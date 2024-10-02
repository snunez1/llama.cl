;;; -*- Mode: LISP; Base: 10; Syntax: ANSI-Common-Lisp; Package: LLAMA2 -*-
;;; Copyright (c) 2023 Andrej
;;; Copyright (c) 2024 Steve Nunez
;;; SPDX-License-identifier: MIT
(in-package #:llama)

;;; Inference for Llama-2 Transformer model in Common Lisp

(defparameter *model* nil)
(defparameter *tokenizer* nil)
(defparameter *sampler* nil)


;; Data structures
(define-binary-struct config ()
  (dim          nil :binary-type u32)	;transformer dimension
  (hidden-dim   nil :binary-type u32)	;for ffn layers
  (num-layers   nil :binary-type u32)	;number of layers (of encoder/decoder blocks)
  (num-heads    nil :binary-type u32)	;number of query heads
  (num-kv-heads nil :binary-type u32)	;number of key/value heads
  (vocab-size   nil :binary-type u32)	;vocabulary size, usually 256 (byte-level)
  (sequence-len nil :binary-type u32))	;max sequence length

(defun print-weights (weights stream depth)
  "TRANSFORMER-WEIGHTS cannot be printed readably"
  (declare (ignore depth))
  (print-unreadable-object (weights stream :type t :identity t) ;let this signal an error if *print-readably* is T
    (princ "" stream)))

(defstruct (transformer-weights (:print-function print-weights))
  token-embedding-table	; vector of length vocab-size, with elements vector of (length dim)

  ;; weights for rmsnorms, vector of vectors
  rms-att-weight	;(layer, dim) rmsnorm weights
  rms-ffn-weight	;(layer, dim)

  ;; weights for matmuls. note dim == num-heads * head-size
  wq			;(layer, dim, number-heads * head-size)
  wk			;(layer, dim, number-kv-heads * head-size)
  wv			;(layer, dim, number-kv-heads * head-size)
  wo			;(layer, number-heads * head-size, dim)

  ;; weights for ffn
  w1			;(layer, hidden-dim, dim)
  w2			;(layer, dim, hidden-dim)
  w3			;(layer, hidden-dim, dim)

  rms-final-weight	;(dim,) final rmsnorm
  wcls)			;(optional) classifier weights for the logits, on the last layer

(defun make-state (config)
  "Allocate buffers for the run state
We technically don't need to do this here, but it may help the compiler generate more efficient code"
  (let+ (((&structure-r/o config- dim hidden-dim num-layers &ign &ign vocab-size sequence-len) config))
	 ;; (kv-dim (/ (* dim num-kv-heads) num-heads))) ;this was in Karpathy's code
    (make-run-state :x   (make-array dim :element-type 'short-float)
		    :xb  (make-array dim :element-type 'short-float)
		    :xb2 (make-array dim :element-type 'short-float)
		    :hb  (make-array hidden-dim :element-type 'short-float)
		    :hb2 (make-array hidden-dim :element-type 'short-float)
		    :q (make-array dim :element-type 'short-float)
		    :k (make-array dim :element-type 'short-float)
		    :v (make-array dim :element-type 'short-float)
		    :attention (make-array sequence-len :element-type 'short-float)
		    :logits    (make-array vocab-size :element-type 'short-float)
		    :key-cache (make-array `(,num-layers ,sequence-len ,dim) :element-type 'short-float)
		    :value-cache (make-array `(,num-layers ,sequence-len ,dim) :element-type 'short-float))))

(defun print-run-state (run-state stream depth)
  "RUN-STATE cannot be printed readably"
  (declare (ignore depth))
  (print-unreadable-object (run-state stream :type t :identity t) ;let this signal an error of *print-readably* is T
    (princ "" stream)))

(defstruct (run-state (:print-function print-run-state))
  "Current wave of activations"
  x		;activation at current time stamp (dim,)
  xb		;same, but inside a residual branch (dim,)
  xb2		;an additional buffer just for convenience (dim,)
  hb		;buffer for hidden dimension in the ffn (hidden_dim,)
  hb2		;buffer for hidden dimension in the ffn (hidden_dim,)
  q		;query (dim,)
  k		;key (dim,)
  v		;value (dim,)
  attention	;buffer for scores/attention values (n_heads, seq_len)
  logits	;output logits

  ;; kv cache
  key-cache	;(layer, seq_len, dim)
  value-cache)	;(layer, seq_len, dim)

(defun print-transformer (transformer stream depth)
  "TRANSFORMER cannot be printed readably"
  (declare (ignore depth))
  (print-unreadable-object (transformer stream :type t :identity t) ;let this signal an error of *print-readably* is T
    (princ "" stream)))

(defstruct (transformer (:print-function print-transformer))
    config  ;the hyperparameters of the architecture (the blueprint)
    weights ;the weights of the model
    state   ;buffers for the "wave" of activations in the forward pass

    ;; some more state needed to properly clean up the memory mapping
    fd				 ;file descriptor for memory mapping
    data			 ;memory mapped data pointer
    file-size)			 ;size of the checkpoint file in bytes


;; TODO: consider LLA:CREATE-ARRAY-FROM-MEMORY and MMAP instead of BINARY-TYPES
;; Also see: https://stackoverflow.com/questions/78992480/binary-foreign-array-to-lisp-array
(defun read-checkpoint (file)
  "Read model checkpoint from file and initialise global variables

Karpathy uses a custom format for these binary models.  There are three versions as of 20240430"
  (let ((binary-types:*endian* :little-endian))
    (binary-types:with-binary-file (stream file :direction :input)
      (let+ ((config (read-binary 'config stream)) ;to return
	     ((&structure-r/o config- dim hidden-dim num-layers num-heads &ign vocab-size sequence-len) config)
	     (head-size (/ dim num-heads))
	     (token-embedding-table (make-array `(,vocab-size ,dim) :element-type 'short-float))
	     (rms-att-weight (make-array num-layers))
	     (rms-ffn-weight (make-array num-layers))

	     ;use an array of 2D arrays so we can access by layer.  This differs from Karpathy's implementation.
	     (wq (make-array num-layers))
	     (wk (make-array num-layers))
	     (wv (make-array num-layers))
	     (wo (make-array num-layers))
	     (w1 (make-array num-layers))
	     (w2 (make-array num-layers))
	     (w3 (make-array num-layers))

	     rms-final-weight
	     (wcls (make-array vocab-size)))

	;; Note these are not in the same order as the structure definition

	(eval `(define-binary-vector token-vector f32 ,dim))
	(eval `(define-binary-array token-array f32 '(,vocab-size ,dim)))
	(setf token-embedding-table (read-binary 'token-array stream))

	(eval `(define-binary-vector rms-weights f32 ,dim))
	(loop for i from 0 below num-layers
	      do (setf (aref rms-att-weight i) (read-binary 'rms-weights stream)))

	(eval `(define-binary-array layer f32 '(,dim ,dim)))
	(loop for i from 0 below num-layers
	      do (setf (aref wq i) (read-binary 'layer stream)))
	(loop for i from 0 below num-layers
	      do (setf (aref wk i) (read-binary 'layer stream)))
	(loop for i from 0 below num-layers
	      do (setf (aref wv i) (read-binary 'layer stream)))
	(loop for i from 0 below num-layers
	      do (setf (aref wo i) (read-binary 'layer stream)))

	;; weights for ffn
	(eval `(define-binary-array w-1&3 f32 '(,hidden-dim ,dim)))
	(eval `(define-binary-array w-2   f32 '(,dim ,hidden-dim)))

	(loop for i from 0 below num-layers
	      do (setf (aref rms-ffn-weight i) (read-binary 'rms-weights stream))) ;rms-weights are same dim
	(loop for i from 0 below num-layers
	      do (setf (aref w1 i) (read-binary 'w-1&3 stream)))
	(loop for i from 0 below num-layers
	      do (setf (aref w2 i) (read-binary 'w-2 stream)))
	(loop for i from 0 below num-layers
	      do (setf (aref w3 i) (read-binary 'w-1&3 stream)))

	(setf rms-final-weight (read-binary 'token-vector stream)) ;same dimension as token-vector

	;; Skip what was previously freq-cis-real & freq-cis-imag (RoPE)
	(loop for i from 0 below (* sequence-len head-size)
	      do (read-binary 'f32 stream))

	(if (> vocab-size 0)
	    (setf wcls token-embedding-table)
	    (setf wcls (read-binary 'token-array stream)))

	(make-transformer :config config
			  :weights (make-transformer-weights :token-embedding-table token-embedding-table
							     :rms-att-weight rms-att-weight
							     :rms-ffn-weight rms-ffn-weight
							     :wq wq
							     :wk wk
							     :wv wv
							     :wo wo
							     :w1 w1
							     :w2 w2
							     :w3 w3
							     :rms-final-weight rms-final-weight
							     :wcls wcls)
			  :state (make-state config))))))

;;; Matrix mathematics

;; If you have LLA loaded, then MM will be automatically imported from
;; there (see pkgdcl.lisp).  There is still room for optimisation with
;; LLA/BLAS.  For example having MM accept an existing output vector
;; instead of malloc and copy results as it does now.

;; If you don't have LLA loaded, the system will use the Common Lisp
;; version of matrix multiplication below.  See the discussions at:
;;   https://gist.github.com/mayerrobert/913b4c26103c614f9517360a4f00286a
;;   http://nklein.com/2009/06/speedy-matrix-multiplication-in-lisp-again/
;;   http://nklein.com/2009/06/trying-to-unconfound-lisp-speeds/
;;   The notes.org file in LLA
;;   http://tkpapp.blogspot.com/2010/05/upgraded-array-element-types-and-pinned.html
;; for ways to optimise this, including SB-SIMD operations.  There is
;; a lot of room for improvement in the Common Lisp matrix multiplication.
#-lla
(defun mm (x y)
  "Multiply vector X with matrix Y using Common Lisp"
  (declare (optimize (compilation-speed 0) (debug 0) (safety 0) (space 0) (speed 3)))
  (declare (type (simple-array single-float 1) x))
  (etypecase y
    ;; Suprisingly, all these perform about the same, despite the first having performance warnings from SBCL
    ;; ((simple-array single-float 1) (aops:vectorize-reduce #'+ (x y) (* x y))) ;3.8 tok/s
    ;; The reason to try MAP is so we can use LPARALLEL:PMAP to speed things up.  It doesn't work, PMAP is slower.
    ((simple-array single-float 1) (reduce #'+ (map 'vector ; 3.6,3.8 tok/s (single thread)
						    #'(lambda (x y)
							(declare (type short-float x y))
								 (* x y))
							x y)))
    (simple-array (aops:each-index* 'single-float (i)
		    (aops:sum-index j
		      (* (aref X j) (aref Y i j)))))))


(defparameter *rms-norm-eps* 1f-05	;1f-6 in Meta's llama2
  "The epsilon used by the rms normalization layers")

(defun rmsnorm (x w)
  "Return the RMS norm of X and scale by weights W"
  (e* x w (/ (sqrt (+ (/ (sum (esquare x)) (length x)) *rms-norm-eps*)))))

(defun softmax (x &optional (size (length x)))
  (let ((max-val (seq-max x))
	sum)
    (loop for i below size
	  do (setf (aref x i) (exp (- (aref x i) max-val)))
	  summing (aref x i) into s
	  finally (setf sum s))
    (e/ x sum)))


(defun forward (token-index position &key (transformer *model*))
  (let+ (((&structure transformer- config weights state) transformer)
	 ((&structure-r/o config- dim &ign num-layers num-heads num-kv-heads &ign &ign) config)
	 ((&structure run-state- x xb xb2 hb hb2 q k v attention logits key-cache value-cache) state)
	 ((&structure transformer-weights-
		      token-embedding-table rms-att-weight rms-ffn-weight wq wk wv wo w1 w2 w3 rms-final-weight wcls)
	  weights)
	 (kv-dim (/ (* dim num-kv-heads) num-heads)) ;Multi Query Attention, see: https://arxiv.org/abs/1911.02150v1
	 ;; (kv-multiplier (/ num-heads num-kv-heads)) ;integer multiplier of the kv sharing in multiquery
	 (head-size (/ dim num-heads)))

    (setf x (aops:sub token-embedding-table token-index))
    (loop for layer below num-layers
	  do (setf xb (rmsnorm x (aref rms-att-weight layer))
	     	   ;; query, key and value matrix multiplications
		   q (mm xb (aref wq layer))
		   k (mm xb (aref wk layer))
		   v (mm xb (aref wv layer)))

	     ;; RoPE relative positional encoding. See: https://arxiv.org/abs/2104.09864
	     ;; You'd think caching the frequency sin/cos vectors would be faster (HF does this), but apparently not:
	     ;; https://github.com/karpathy/llama2.c/issues/302
	     (loop for i below dim by 2
		   for head-dim = (mod i head-size)
		   for freq     = (/ (expt 10000f0 (/ head-dim head-size)))
		   for val      = (* position freq)
		   for fcr      = (cos val)
		   for fci      = (sin val)
		   for rotn     = (if (< i kv-dim) 2 1) ;how many vectors? 2 = q & k, 1 = q only
		   do (loop for v below rotn
			    for vec = (if (= v 0) q k) ;the vector to rotate, query or key
			    for v0  = (aref vec i)
			    for v1  = (aref vec (1+ i))
			    do (setf (aref vec i) (- (* v0 fcr) (* v1 fci))
				     (aref vec (1+ i)) (+ (* v0 fci) (* v1 fcr)))))

	     ;; Save key and value at this timestep (position) in cache
	     (setf (sub key-cache   layer position) (copy-array k)
		   (sub value-cache layer position) (copy-array v))

	     ;; Multiquery attention, iterate over all heads
	     ;; (lparallel:pdotimes (head num-heads) ; raises floating point errors
	     (dotimes (head num-heads)	;TODO: make multi-threaded
	       (loop for timestep upto position
		     for sqrt-head-size = (sqrt head-size)
		     for head-q = (subseq q (* head head-size) (* (1+ head) head-size))
		     for head-k = (subseq
				   (sub key-cache layer timestep) (* head head-size) (* (1+ head) head-size))
		     do (setf (aref attention timestep) (/ (mm head-q head-k) sqrt-head-size)))

	       (setf attention (softmax attention (1+ position)))

	       ;; weighted sum of the values, store back into xb
	       (let ((xb (partition xb (* head head-size) (* (1+ head) head-size))))
		 (aops:each-index (i)
		   (setf (aref xb i) 0.0))
		 (loop for timestep upto position
		       for att = (aref attention timestep)
		       for   v = (partition (sub value-cache layer timestep) (* head head-size) (* (1+ head) head-size))
		       do (loop for i below head-size
				do (incf (aref xb i) (* att (aref v i)))))))

	     (setf xb2 (mm xb (aref wo layer)) ;final matmul to get the output of the attention
		   x   (e+ x xb2)	       ;residual connection back into x
		   xb  (rmsnorm x (aref rms-ffn-weight layer)) ;ffn rms norm
		   hb  (mm xb (aref w1 layer))
		   hb2 (mm xb (aref w3 layer))
		   hb  (e* hb (e/ (e* (e+ 1 (eexp (e- hb)))))) ;silu(x)=x*σ(x), where σ(x) is the logistic sigmoid
		   hb  (e* hb hb2)			       ;elementwise multiply with w3(x)
		   xb  (mm hb (aref w2 layer))		       ;final matmul to get the output of the ffn
		   x   (e+ x xb)))			       ;residual connection
    ;; Layer loop ends above this line

    (setf x      (rmsnorm x rms-final-weight)    ;final rms norm
	  logits (mm x token-embedding-table)))) ;classifier into logits


;;; Tokenizer
(defun print-tokenizer (tokenizer stream depth)
  "TOKENIZER cannot be printed readably"
  (declare (ignore depth))
  (print-unreadable-object (tokenizer stream :type t :identity t) ;let this signal an error of *print-readably* is T
    (princ "" stream)))

(defstruct (tokenizer (:print-function print-tokenizer))
  vocabulary
  vocabulary-scores
  vocabulary-size
  max-token-length)

(defun make-vocabulary (file vocabulary-size)
  (let ((vocabulary        (make-array vocabulary-size :element-type 'string))
	(vocabulary-scores (make-array vocabulary-size :element-type 'float))
	max-token-length)
    (mmap:with-mmap (addr fd size file)
      (setf max-token-length (cffi:mem-ref addr :int))
      (loop for i below vocabulary-size
	    for ptr = (cffi:inc-pointer addr 4) then (cffi:inc-pointer ptr (+ 4 4 count))
	    for score = (cffi:mem-ref ptr :float)
	    for count = (cffi:mem-ref ptr :int 4)
	    for token = (cffi:foreign-string-to-lisp ptr :offset 8 :count count)
	    do (setf (aref vocabulary i) token
		     (aref vocabulary-scores i) score)
	    finally (return (values vocabulary vocabulary-scores max-token-length))))))

(defun encode (text vocabulary scores)
  (let ((tokens (map 'vector (lambda (c) (position c vocabulary :test #'string=)) text)))
    (loop named outer
	  for best-score = -1e10
	  for best-id = -1
	  for best-index = -1
	  do (loop for i below (1- (length tokens))
		   for string = (concatenate 'string
					     (aref vocabulary (aref tokens i))
					     (aref vocabulary (aref tokens (1+ i))))
		   for id = (position string vocabulary :test #'string=)
		   if (and id (> (aref scores id) best-score)) ;This merge pair exists in vocabulary
		     do (setf best-score (aref scores id)
			      best-id id
			      best-index i))

	     (if (= best-index -1) (return-from outer tokens))
	     (setf (aref tokens best-index) best-id
		   tokens (concatenate 'vector (subseq tokens 0 (1+ best-index))
				               (subseq tokens (+ 2 best-index)))))))

;;; Sampler - greedy argmax, random, top-p, top-k
;;; Takes logits and returns a sampled token

(defun sample-mult (logits)
  (let ((r (random 1.0)))
    (loop for i below (length logits)
	  summing (aref logits i) into cdf
	  if (< r cdf) return i
	  finally (return (1- (length logits))))))

(defun sort-scores (scores predicate)
  "Returns an array of CONS, (index . score), sorted by score)."
  (let ((index -1))
    (sort (map 'vector (lambda (x)
			 (cons (incf index) x))
	       scores)
	  predicate :key #'cdr)))

;; I suspect that Karpathy's implementation takes the code path for
;; rounding errors.  Removing all scores below threshold, at least the
;; first time through, results in an empty set.
(defun sample-topp (logits p)
  (let* (;;(cutoff (/ (- 1.0 p) (- (length logits) 1)))	;values smaller than this cannot be part of the result
	 ;; (probabilities (sort-scores (remove-if #'(lambda (x) (< x cutoff)) logits) #'>)) ;remove smaller than cutoff and sort result
	 (probabilities (sort-scores logits #'>))
	 (r (random 1.0))
	 (last-index))

	(setf last-index (loop for i below (length probabilities)
			       summing (cdr (aref probabilities i)) into cumulative-probability
			       if (> cumulative-probability p) return i
			       finally (return  (1- (length logits)))))

	;; Sample from our truncated sequence
	(loop for i below (length (subseq probabilities 0 last-index))
	      summing (cdr (aref probabilities i)) into cdf
	      if (< r cdf) return (car (aref probabilities i))
	      finally (return (car (aref probabilities last-index))))))


(defun sample (logits temperature &key topp topk)
  (declare (ignore topk))
  (if (< temperature short-float-epsilon)
      (argmax logits)
      (progn
	(setf logits (e/ logits temperature)
	      logits (softmax logits))
	(if (or (null topp) (<= topp 0) (>= topp 1))
	    (sample-mult logits)
	    (sample-topp logits topp)))))


;;;
;;; User API below
;;;
(defun init (model-path tokenizer-path &optional vocabulary-size) ;TODO: default to files in the repo
  "Initialise the model and tokenizer"
  (let+ (((&values vocabulary scores max-token-length) (make-vocabulary tokenizer-path vocabulary-size)))
    (setf *model*     (read-checkpoint model-path)
	  *tokenizer* (make-tokenizer :vocabulary vocabulary
				      :vocabulary-scores scores
				      :vocabulary-size vocabulary-size
				      :max-token-length max-token-length)))
  (values))


;;; Generation
(defun generate (model tokenizer &key
				   topp
				   (temperature 0.9)
				   (steps 256)
				   prompt)
  (let+ (((&structure tokenizer- vocabulary vocabulary-scores vocabulary-size max-token-length) tokenizer)
	 (token 1) next-token
	 (prompt-tokens (encode prompt vocabulary vocabulary-scores))
	 (start-time (get-internal-real-time)) end-time)

    (unlessf prompt (aref vocabulary 1)) ;BoS

    (loop for position below steps
	  for logits = (forward token position :transformer model)
	  do (if (< position (length prompt-tokens))
		 (setf next-token (aref prompt-tokens position))
		 (setf next-token (sample logits temperature :topp topp)))
	     (format t "~A" (aref vocabulary next-token))
	     (setf token next-token))

    (setf end-time (get-internal-real-time))
    (let ((tok/s (float (/ steps (/ (- end-time start-time) internal-time-units-per-second)))))
      (format t "~%tokens/s: ~A~%" tok/s)
      tok/s)))
