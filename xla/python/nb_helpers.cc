/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/python/nb_helpers.h"

#include <Python.h>

#include "nanobind/nanobind.h"  // from @nanobind

namespace nb = nanobind;

namespace xla {

ssize_t nb_hash(nb::handle o) {
  Py_hash_t h = PyObject_Hash(o.ptr());
  if (h == -1) {
    throw nb::python_error();
  }
  return h;
}

}  // namespace xla
