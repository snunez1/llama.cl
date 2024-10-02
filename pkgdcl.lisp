;;; -*- Mode: LISP; Base: 10; Syntax: ANSI-Common-Lisp; Package: CL-USER -*-
;;; Copyright (c) 2024 Steve Nunez
;;; SPDX-License-identifier: MIT

(uiop:define-package "LLAMA"
  (:use #:cl #:let-plus #:num-utils.elementwise)
  (:import-from :binary-types #:define-binary-struct #:define-binary-vector #:define-binary-array
		              #:read-binary #:u8 #:u32 #:f32)
  (:import-from #:num-utils.arithmetic #:sum #:seq-max)
  (:import-from #:alexandria #:copy-array)
  (:import-from #:array-operations #:partition #:sub #:argmax)
  #+lla (:import-from #:lla #:mm)
  (:import-from #:alexandria+ #:unlessf)
  (:export #:read-checkpoint #:make-vocabulary #:generate))
