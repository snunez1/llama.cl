;;; -*- Mode: LISP; Base: 10; Syntax: ANSI-Common-Lisp; Package: CL-USER -*-
;;; Copyright (c) 2023 Andrej
;;; Copyright (c) 2024 Symbolics Pte Ltd
;;; SPDX-License-identifier: MIT

(defsystem "llama"
  :version "0.0.1"
  :license :MIT
  :author "Steve Nunez <steve@symbolics.tech>"
  :long-name   "Llama for Common Lisp"
  :description "Llama for Common Lisp"
  :long-description "A port of Karparty's llama2 inference code to Common Lisp"
  :source-control (:git "https://github.com/snunez1/llama.cl.git")
  :bug-tracker "https://github.com/snunez1/llama.cl/issues"
  :depends-on ("num-utils" "array-operations" "alexandria" "alexandria+" "let-plus" "binary-types" "mmap")
  :components ((:file "pkgdcl")
	       (:file "run" :depends-on ("pkgdcl"))))
