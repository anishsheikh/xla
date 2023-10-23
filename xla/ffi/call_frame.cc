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

#include "xla/ffi/call_frame.h"

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "xla/ffi/c/c_api.h"
#include "xla/ffi/c/c_internal_api.h"  // IWYU pragma: keep
#include "xla/stream_executor/device_memory.h"
#include "xla/xla_data.pb.h"

namespace xla::ffi {

//===----------------------------------------------------------------------===//
// CallFrameBuilder
//===----------------------------------------------------------------------===//

void CallFrameBuilder::AddBufferArg(se::DeviceMemoryBase memory,
                                    PrimitiveType type,
                                    absl::Span<const int64_t> dims) {
  args_.push_back(BufferArg{memory, type, {dims.begin(), dims.end()}});
}

void CallFrameBuilder::AddI32Attr(std::string name, int32_t value) {
  attrs_.try_emplace(std::move(name), value);
}

void CallFrameBuilder::AddF32Attr(std::string name, float value) {
  attrs_.try_emplace(std::move(name), value);
}

void CallFrameBuilder::AddStringAttr(std::string name, std::string value) {
  attrs_.try_emplace(std::move(name), value);
}

CallFrame CallFrameBuilder::Build(XLA_FFI_Api* api,
                                  XLA_FFI_ExecutionContext* ctx) {
  return CallFrame(api, ctx, args_, attrs_);
}

//===----------------------------------------------------------------------===//
// CallFrame
//===----------------------------------------------------------------------===//

CallFrame::CallFrame(XLA_FFI_Api* api, XLA_FFI_ExecutionContext* ctx,
                     absl::Span<const CallFrameBuilder::BufferArg> args,
                     const CallFrameBuilder::AttributesMap& attrs)
    : arguments_(InitArgs(args)), attributes_(InitAttrs(attrs)) {
  call_frame_.api = api;
  call_frame_.ctx = ctx;
  call_frame_.args = arguments_.ffi_args;
  call_frame_.attrs = attributes_.ffi_attrs;
}

//===----------------------------------------------------------------------===//
// Call frame arguments
//===----------------------------------------------------------------------===//

/*static*/ CallFrame::Arguments CallFrame::InitArgs(
    absl::Span<const CallFrameBuilder::BufferArg> args) {
  Arguments res;

  // Convert call frame builder arguments to call frame arguments.
  for (auto& arg : args) {
    CallFrame::BufferArg buffer;
    buffer.dims = arg.dims;
    buffer.buffer.data = const_cast<void*>(arg.memory.opaque());
    buffer.buffer.primitive_type = static_cast<uint8_t>(arg.type);
    buffer.buffer.rank = buffer.dims.size();
    res.arguments.push_back(std::move(buffer));
  }

  // Fixup pointers in XLA FFI structs.
  for (auto& arg : res.arguments) {
    arg.buffer.dims = arg.dims.data();
  }

  // Initialize vectors required for building XLA_FFI_Args.
  for (auto& arg : res.arguments) {
    res.types.push_back(XLA_FFI_ArgType_BUFFER);
    res.args.push_back(&arg.buffer);
  }

  // Finally initialize XLA FFI struct. At this point all storage is allocated
  // and it's safe to grab a pointer to it.
  res.ffi_args.num_args = res.arguments.size();
  res.ffi_args.types = res.types.data();
  res.ffi_args.args = res.args.data();

  return res;
}

//===----------------------------------------------------------------------===//
// Call frame attributes
//===----------------------------------------------------------------------===//

/*static*/ void CallFrame::FixupString(CallFrame::String& str) {
  str.span.data = str.value.data();
  str.span.size = str.value.size();
}

// An std::visit overload set for converting CallFrameBuilder::Attribute to
// CallFrame::Attribute.
struct CallFrame::ConvertAttribute {
  template <typename T>
  CallFrame::Attribute operator()(const T& value) {
    return value;
  }

  CallFrame::Attribute operator()(const std::string& str) {
    return CallFrame::String{str};
  }
};

// An std::visit overload set to fixup CallFrame::Attribute storage and
// initialize XLA FFI structs with valid pointers into storage objects.
struct CallFrame::FixupAttribute {
  template <typename T>
  void operator()(T& value) {}

  void operator()(CallFrame::String& str) { FixupString(str); }
};

// An std::visit overload set to get CallFrame::Attribute XLA FFI type.
struct CallFrame::AttributeType {
  XLA_FFI_AttrType operator()(int32_t&) { return XLA_FFI_AttrType_I32; }

  XLA_FFI_AttrType operator()(float&) { return XLA_FFI_AttrType_F32; }

  XLA_FFI_AttrType operator()(CallFrame::String&) {
    return XLA_FFI_AttrType_STRING;
  }
};

// An std::visit overload set to get CallFrame::Attribute storage pointer.
struct CallFrame::AttributeStorage {
  template <typename T>
  void* operator()(T& value) {
    return &value;
  }

  void* operator()(CallFrame::String& str) { return &str.span; }
};

/*static*/ CallFrame::Attributes CallFrame::InitAttrs(
    const CallFrameBuilder::AttributesMap& attrs) {
  Attributes res;

  // Convert call frame builder attributes to a collections of named attributes.
  for (auto& [name, attr] : attrs) {
    NamedAttribute named = {String{name}, std::visit(ConvertAttribute(), attr)};
    res.attributes.push_back(std::move(named));
  }

  // Sort attributes by name to enable binary search at run time.
  absl::c_sort(res.attributes, [](const auto& a, const auto& b) {
    return a.name.value < b.name.value;
  });

  // Fixup XLA FFI structs to point to correct storage.
  for (auto& attr : res.attributes) {
    FixupString(attr.name);
    std::visit(FixupAttribute{}, attr.value);
  }

  // Initialize vectors required for building XLA_FFI_Attributes.
  for (auto& attr : res.attributes) {
    res.names.push_back(&attr.name.span);
    res.types.push_back(std::visit(AttributeType(), attr.value));
    res.attrs.push_back(std::visit(AttributeStorage(), attr.value));
  }

  // Finally initialize XLA FFI struct. At this point all storage is allocated
  // and it's safe to grab a pointer to it.
  res.ffi_attrs.num_attrs = res.attributes.size();
  res.ffi_attrs.names = res.names.data();
  res.ffi_attrs.types = res.types.data();
  res.ffi_attrs.attrs = res.attrs.data();

  return res;
}

}  // namespace xla::ffi
