# XLA FFI

https://en.wikipedia.org/wiki/Foreign_function_interface

```
A foreign function interface (FFI) is a mechanism by which a program written in
one programming language can call routines or make use of services written or
compiled in another one. An FFI is often used in contexts where calls are made
into binary dynamic-link library.
```

XLA FFI is a mechanism by which an XLA program can call functions compiled with
another programming language using a stable C API (which guarantees ABI
compatibility between XLA and external functions) and a C++ header only library
that hides all the details of underlying C API from the user.

This is the next generation of XLA custom calls with a rich type safe APIs.

**WARNING:** Under construction. We have an rich type safe custom call mechanism
for XLA runtime, however it's not providing any stable C API, this is an attempt
to replicate the usability of XLA runtime custom calls with a stable C API.