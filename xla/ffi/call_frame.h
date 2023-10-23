/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_FFI_CALL_FRAME_H_
#define XLA_FFI_CALL_FRAME_H_

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xla/ffi/c/c_api.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/types.h"  // IWYU pragma: keep
#include "xla/xla_data.pb.h"

namespace xla::ffi {

// CallFrame library encodes C++ arguments using XLA FFI C API structs in a form
// compatible with a call frame decoding defined by `api.h` header file.

//===----------------------------------------------------------------------===//
// CallFrameBuilder
//===----------------------------------------------------------------------===//

class CallFrame;  // forward declare

class CallFrameBuilder {
 public:
  CallFrame Build(XLA_FFI_Api* api, XLA_FFI_ExecutionContext* ctx);

  void AddBufferArg(se::DeviceMemoryBase memory, PrimitiveType type,
                    absl::Span<const int64_t> dims);

  void AddI32Attr(std::string name, int32_t value);
  void AddF32Attr(std::string name, float value);
  void AddStringAttr(std::string name, std::string value);

 private:
  friend class CallFrame;

  using Attribute = std::variant<int32_t, float, std::string>;
  using AttributesMap = absl::flat_hash_map<std::string, Attribute>;

  struct BufferArg {
    se::DeviceMemoryBase memory;
    PrimitiveType type;
    std::vector<int64_t> dims;
  };

  std::vector<BufferArg> args_;
  AttributesMap attrs_;
};

//===----------------------------------------------------------------------===//
// CallFrame
//===----------------------------------------------------------------------===//

class CallFrame {
 public:
  const XLA_FFI_CallFrame* call_frame() const { return &call_frame_; }

 private:
  friend class CallFrameBuilder;

  // Declare implementation detail structs to grant access to private members.
  struct ConvertAttribute;
  struct FixupAttribute;
  struct AttributeType;
  struct AttributeStorage;

  //--------------------------------------------------------------------------//
  // Call frame arguments
  //--------------------------------------------------------------------------//

  // BufferArg combines a storage for buffer dimensions and an XLA FFI struct
  // defining a buffer argument. It is a user responsibility to update `buffer`
  // member to point to `dims` storage.
  struct BufferArg {
    std::vector<int64_t> dims;
    XLA_FFI_Buffer buffer = {XLA_FFI_Buffer_STRUCT_SIZE, nullptr};
  };

  struct Arguments {
    std::vector<BufferArg> arguments;

    std::vector<XLA_FFI_ArgType> types;  // XLA_FFI_Args::types
    std::vector<void*> args;             // XLA_FFI_Args::args

    XLA_FFI_Args ffi_args = {XLA_FFI_Args_STRUCT_SIZE, nullptr};
  };

  //--------------------------------------------------------------------------//
  // Call frame attributes
  //--------------------------------------------------------------------------//

  struct String;  // forward declare

  using Attribute = std::variant<int32_t, float, String>;

  // String combines a string storage and an XLA FFI byte span referencing it.
  // It is a user responsibility to update the `span` member to point to the
  // `value` storage. It should be done with extra care, e.g. when String added
  // to `std::vector` container, `span` will be invalidated by vector
  // reallocation when it grows.
  struct String {
    std::string value;
    XLA_FFI_ByteSpan span = {XLA_FFI_ByteSpan_STRUCT_SIZE, nullptr};
  };

  struct NamedAttribute {
    String name;
    Attribute value;
  };

  struct Attributes {
    std::vector<NamedAttribute> attributes;

    std::vector<XLA_FFI_ByteSpan*> names;  // XLA_FFI_Attributes::names
    std::vector<XLA_FFI_AttrType> types;   // XLA_FFI_Attributes::types
    std::vector<void*> attrs;              // XLA_FFI_Attributes::attrs

    XLA_FFI_Attrs ffi_attrs = {XLA_FFI_Attrs_STRUCT_SIZE, nullptr};
  };

  //--------------------------------------------------------------------------//

  CallFrame(XLA_FFI_Api* api, XLA_FFI_ExecutionContext* ctx,
            absl::Span<const CallFrameBuilder::BufferArg> args,
            const CallFrameBuilder::AttributesMap& attrs);

  static Arguments InitArgs(absl::Span<const CallFrameBuilder::BufferArg> args);
  static Attributes InitAttrs(const CallFrameBuilder::AttributesMap& attrs);

  static void FixupString(CallFrame::String& str);

  Arguments arguments_;
  Attributes attributes_;

  XLA_FFI_CallFrame call_frame_ = {XLA_FFI_CallFrame_STRUCT_SIZE, nullptr};
};

}  // namespace xla::ffi

#endif  // XLA_FFI_CALL_FRAME_H_
