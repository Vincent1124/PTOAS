# PTOAS `_core.so` 解耦操作流程

## 目的与范围

把 pybind11 绑定实现 `PTOModule.cpp` 从共享编译器 DSO `libPTOASCompiler` 移回 `_core` 扩展，使 `libPTOASCompiler` 变成**与 CPython 版本无关**、可跨多个 Python ABI 共享的一份。

解耦的验收基线是：解耦前后同一个 Python 版本下，`import ptoas._core`、`register_dialect`、各类型/属性构造、端到端编译用例的行为必须一致。

### 为什么必须解耦

`PTOModule.cpp` 是 pybind11 翻译单元，编译时绑定到具体 `Python.h` 的 CPython ABI（结构体布局、宏、内联函数）。只要它编进 `libPTOASCompiler`，DSO 就与某个 Python 版本绑定，无法跨版本共享。把它移回 `_core` 后，DSO 侧不再残留任何 CPython 符号。

代价是 `_core` 承接 `PTOModule.cpp` 后，需要从 DSO 解析一批 PTO 类型符号（本身这类类型符号与 `PTOModule.cpp` 一起编译进 `libPTOASCompiler`，处于 hidden 状态，现在跨库链接需要将这批符号进行对外暴露）。这批符号是**纯 C++ MLIR/PTO 符号，与 CPython ABI 无关**，因此是一次性、跨所有版本共享的导出面，不会重新引入版本耦合。方案采用**收敛的 C API 化**：把 `PTOModule.cpp` 里仅剩的 C++ 直调类型全部改走 C API，DSO 导出面保持「C API + 桥接函数」，最干净最稳定。

## 当前状态（main 分支，已核实）

关键文件与行号（main 分支实测，与旧设计文档路径不同，以本节为准）：

- `tools/ptoas/CMakeLists.txt`
  - `:118` `"${CMAKE_SOURCE_DIR}/lib/Bindings/Python/PTOModule.cpp"` 在 `PTOASCompilerImplementation`（DSO 聚合根）的 `SOURCES` 中。
  - `:131` `target_link_libraries(obj.PTOASCompilerImplementation PRIVATE pybind11::module)`。
  - `:136-142` 为 `PTOModule.cpp` 打的 per-source `-frtti -fexceptions` / `/EHsc /GR`。
  - `:146` `target_link_libraries(PTOASCompiler PRIVATE pybind11::module)`。
  - `:147-149` Linux 下 `target_link_options(PTOASCompiler PRIVATE "LINKER:-z,undefs")`。
  - `:150-153` DSO 的 `CXX_VISIBILITY_PRESET hidden` + `VISIBILITY_INLINES_HIDDEN`。
  - `:158-166` `add_mlir_python_extension(PTOASPythonCore _core ... SOURCES NativeModule.cpp LINK_LIBS PTOASCompiler)`。
  - `:170-172` `PTOASPythonCore` 已 include `lib/Bindings/Python`。
- `lib/Bindings/Python/PTOModule.cpp`（1555 行）——待迁移到 `_core`。
- `tools/ptoas/NativeModule.cpp`（207 行）——已在 `_core`，不动。
- `include/pto-c/Dialect/PTOCAPI.h`（273 行）——C API 声明。
- `lib/CAPI/Dialect/PTO.cpp`（930 行）——C API 实现。

### `PTOModule.cpp` 里仍走 C++ 直调的 7 处类型（需 C API 化）

`PTOModule.cpp` 大部分类型/属性已走 C API（`mlirPTO*`）。仅剩以下 7 类经由 C++ `mlir::pto::XxxType::get(...)` / `cast<>` / `isa<>` 使用（行号为 main 实测）：

| 类型 | 行号 | 用到的 C++ 方法 |
|------|------|------|
| `VRegType` | 1020-1042 | `get`、`getElementCount`、`getElementType`、`isa` |
| `MaskType` | 1046-1060 | `get`、`getGranularity`、`isa` |
| `VMIVRegType` | 1064-1101 | `get`、`getElementCount`、`getElementType`、`getLayout`、`isa` |
| `VMIMaskType` | 1103-1140 | `get`、`getElementCount`、`getGranularity`、`getLayout`、`isa` |
| `AlignType` | 1142-1150 | `get`、`isa` |
| `StructType` | 1153-1195 | `getChecked`（含 `mlir::emitError`/`mlir::UnknownLoc::get`）、`getFieldTypes`、`isa` |
| `TileBufType` getters | 1451-1553 | `getRank`/`getElementType`/`getMemorySpace`/`getShape`/`getValidShape`/`getBLayoutAttr`/`getSLayoutAttr`/`getBLayoutValueI32`/`getSLayoutValueI32`/`getPadValueI32`/`getCompactModeI32`/`getSFractalSizeI32`（构造 `Get`/`IsA` 已有 C API） |

`mlir::pto::*` 枚举（`AddressSpace`/`BLayout`/... 等）来自头文件、编译期常量，**无需**导出符号。`ShapedType::kDynamic` 为 `constexpr`，同样无需导出。

### C API 导出面缺口（易漏，必须先补）

`PTOModule.cpp` 现在编在 DSO 内部，`mlirPTO*` 符号内部解析即可，因此 `include/pto-c/Dialect/PTOCAPI.h` 的**早期声明块（`:27-89`）未标注 `MLIR_CAPI_EXPORTED`**：PtrType、Async*、HiF8 系列、AddressSpace、TensorView、PartitionTensorView、Tile。已核实 `PTOModule.cpp` **全部调用**了这些声明。解耦后这些符号必须跨 DSO 边界被 `_core` 解析；在 DSO 的 `hidden` 可见性下它们会被隐藏，导致 `_core` 链接时 undefined symbol。

（`:92` 起的 TileBufType/各 Attr 声明已带 `MLIR_CAPI_EXPORTED`，无需处理。）

## 操作步骤

严格按顺序执行，先补 C API、再改绑定、最后动 CMake。每步的失败模式见「验证」。

### 步骤 1：给早期 C API 声明补 `MLIR_CAPI_EXPORTED`

文件 `include/pto-c/Dialect/PTOCAPI.h`。给 `:27-89` 范围内所有函数声明加 `MLIR_CAPI_EXPORTED`（与 `:92` 起 TileBufType 块的写法一致）。覆盖：

- PtrType：`:27-32`
- Async session/event/prefetch：`:35-40`
- HiF8/F8E8M0/HiF8x2/F4E1M2x2/F4E2M1x2/BF16x2：`:43-54`
- AddressSpace：`:57-63`
- TensorView：`:66-72`
- PartitionTensorView：`:75-81`
- Tile：`:84-89`

### 步骤 2：在 `PTO.h` 新增 35 个 C API 声明

在 `extern "C"` 块内新增以下声明，全部加 `MLIR_CAPI_EXPORTED`。对应 7 处 C++ 直调类型。签名如下：

```c
// ---- !pto.vreg<count x elem> ----
bool     mlirPTOTypeIsAVRegType(MlirType type);
MlirType mlirPTOVRegTypeGet(MlirContext ctx, int64_t elementCount, MlirType elementType);
int64_t  mlirPTOVRegTypeGetElementCount(MlirType type);
MlirType mlirPTOVRegTypeGetElementType(MlirType type);

// ---- !pto.mask<granularity> ----
bool          mlirPTOTypeIsAMaskType(MlirType type);
MlirType      mlirPTOMaskTypeGet(MlirContext ctx, MlirStringRef granularity);
MlirStringRef mlirPTOMaskTypeGetGranularity(MlirType type);

// ---- !pto.vmivreg<count x elem, layout?> ----
bool          mlirPTOTypeIsAVMIVRegType(MlirType type);
MlirType      mlirPTOVMIVRegTypeGet(MlirContext ctx, int64_t elementCount,
                                    MlirType elementType, MlirAttribute layout /*may be null*/);
int64_t       mlirPTOVMIVRegTypeGetElementCount(MlirType type);
MlirType      mlirPTOVMIVRegTypeGetElementType(MlirType type);
MlirAttribute mlirPTOVMIVRegTypeGetLayout(MlirType type); // null attr if absent

// ---- !pto.vmimask<count x granularity, layout?> ----
bool          mlirPTOTypeIsAVMIMaskType(MlirType type);
MlirType      mlirPTOVMIMaskTypeGet(MlirContext ctx, int64_t elementCount,
                                    MlirStringRef granularity, MlirAttribute layout /*may be null*/);
int64_t       mlirPTOVMIMaskTypeGetElementCount(MlirType type);
MlirStringRef mlirPTOVMIMaskTypeGetGranularity(MlirType type);
MlirAttribute mlirPTOVMIMaskTypeGetLayout(MlirType type); // null attr if absent

// ---- !pto.align ----
bool     mlirPTOTypeIsAAlignType(MlirType type);
MlirType mlirPTOAlignTypeGet(MlirContext ctx);

// ---- !pto.struct<fields...> ----
bool     mlirPTOTypeIsAStructType(MlirType type);
// Null type (mlirTypeIsNull) if field types are invalid; emits a diagnostic
// on an UnknownLoc, mirroring StructType::getChecked.
MlirType mlirPTOStructTypeGet(MlirContext ctx, intptr_t numFieldTypes,
                              MlirType const *fieldTypes);
intptr_t mlirPTOStructTypeGetNumFieldTypes(MlirType type);
MlirType mlirPTOStructTypeGetFieldType(MlirType type, intptr_t index);

// ---- TileBufType getters（Get 已有 CAPI，仅补属性读取） ----
intptr_t       mlirPTOTileBufTypeGetRank(MlirType type);
MlirType       mlirPTOTileBufTypeGetElementType(MlirType type);
MlirAttribute  mlirPTOTileBufTypeGetMemorySpace(MlirType type);
const int64_t *mlirPTOTileBufTypeGetShape(MlirType type, intptr_t *numDimsOut);
const int64_t *mlirPTOTileBufTypeGetValidShape(MlirType type, intptr_t *numDimsOut);
MlirAttribute  mlirPTOTileBufTypeGetBLayoutAttr(MlirType type);
MlirAttribute  mlirPTOTileBufTypeGetSLayoutAttr(MlirType type);
int32_t        mlirPTOTileBufTypeGetBLayoutValue(MlirType type);
int32_t        mlirPTOTileBufTypeGetSLayoutValue(MlirType type);
int32_t        mlirPTOTileBufTypeGetPadValue(MlirType type);
int32_t        mlirPTOTileBufTypeGetCompactMode(MlirType type);
int32_t        mlirPTOTileBufTypeGetSFractalSize(MlirType type);
```

### 步骤 3：在 `PTO.cpp` 实现这 35 个函数

文件 `lib/CAPI/Dialect/PTO.cpp`。照既有 `TensorViewType`/`TileBufType` 写法：`unwrap`/`wrap` + `cast<mlir::pto::XxxType>`。底层 C++ 方法（已在 `PTOModule.cpp` 用到，确认存在）：

- `VRegType`/`VMIVRegType`：`getElementCount()`、`getElementType()`
- `MaskType`/`VMIMaskType`：`getGranularity()`
- `VMIVRegType`/`VMIMaskType`：`getLayout()`
- `StructType`：`getChecked(...)`、`getFieldTypes()`
- `TileBufType`：`getRank()`、`getElementType()`、`getMemorySpace()`、`getShape()`、`getValidShape()`、`getBLayoutAttr()`、`getSLayoutAttr()`、`getBLayoutValueI32()`、`getSLayoutValueI32()`、`getPadValueI32()`、`getCompactModeI32()`、`getSFractalSizeI32()`

要点：

- `mlirPTOStructTypeGet` 用 `StructType::getChecked`，需要一个诊断 loc；照 `PTOModule.cpp` 原逻辑用 `mlir::emitError(mlir::UnknownLoc::get(ctx))`。**这些上游 MLIR C++ 调用全部封在 CAPI 实现内**，构造失败返回 null type（`_core` 侧用 `mlirTypeIsNull` 判定并抛 `py::value_error`）。
- shape/validShape getter 返回内部数组指针（只读）+ `numDimsOut`，与既有 `mlirPTOTensorViewTypeGetShape` 一致。
- `MlirStringRef` 的返回：granularity 用 `wrap(StringRef)`；`_core` 侧再 `.str()`。

### 步骤 4：改写 `PTOModule.cpp` 的 7 处类型子类到 C API

文件 `lib/Bindings/Python/PTOModule.cpp`。把 1020-1195、1451-1553 的 `mlir::pto::XxxType::get/cast/isa` 全部替换为步骤 2 的 C API 调用。改完后 `PTOModule.cpp` **不再触碰任何上游 MLIR C++ 类型符号**（`grep -nE 'mlir::pto::[A-Z].*Type|::getChecked|mlir::emitError|mlir::UnknownLoc' lib/Bindings/Python/PTOModule.cpp` 应只剩枚举用法）。

- `isa<mlir::pto::VRegType>(unwrap(type))` → `mlirPTOTypeIsAVRegType(type)`
- `VRegType::get(...)` → `mlirPTOVRegTypeGet(context, count, elem)`
- `cast<...>(unwrap(self)).getElementCount()` → `mlirPTOVRegTypeGetElementCount(self)`
- StructType 的 null 判定改用 `mlirTypeIsNull`。
- TileBufType getters 逐一替换为 `mlirPTOTileBufType*` C API。

### 步骤 5：CMake 迁移

文件 `tools/ptoas/CMakeLists.txt`。

1. **迁移源文件**：把 `:118` 的 `PTOModule.cpp` 从 `PTOASCompilerImplementation` 的 `SOURCES` 删除；加入 `add_mlir_python_extension(PTOASPythonCore _core ...)` 的 `SOURCES`（`:163`）与 `NativeModule.cpp` 并列。
2. **迁移 pybind11 链接契约**：删除 `:131` `target_link_libraries(obj.PTOASCompilerImplementation PRIVATE pybind11::module)` 与 `:146` `target_link_libraries(PTOASCompiler PRIVATE pybind11::module)`。`_core` 走 `add_mlir_python_extension` 自带 pybind11 模块链接契约。
3. **撤销为 pybind11 打的 DSO 例外**：
   - 删除 `:147-149` 的 `-z,undefs`。DSO 不再含未定义 CPython 符号，恢复 `add_mlir_aggregate` 默认的 `-z defs`。
   - 删除 `:136-142` 对 `PTOModule.cpp` 的 per-source `-frtti -fexceptions` / `/EHsc /GR`——该 TU 现编进 `_core`，RTTI/异常由扩展常规编译选项提供。
4. **保留** `:150-153` DSO 的 `hidden` 可见性（导出面靠 `MLIR_CAPI_EXPORTED` 标注控制，与本次 C API 化配套）。
5. `_core` 的 include：`:170-172` 已有 `lib/Bindings/Python`；`PTOModule.h` 在该目录下，确认可被 `_core` 找到。

## 验证（单版本行为等价）

1. **配置/编译**：`cmake --build`（或 `./build.sh`）通过。
   - 失败模式 A（漏了步骤 1）：`_core` 链接报 `undefined symbol: mlirPTO*`（PtrType/Async/HiF8/AddressSpace/TensorView/Tile 之一）→ 回到步骤 1 补 `MLIR_CAPI_EXPORTED`。
   - 失败模式 B（漏了步骤 4 某处）：`_core` 链接报 `undefined symbol: mlir::pto::XxxType::...` 或 `mlir::emitError` → 该处仍在走 C++ 直调，回步骤 4。
2. **DSO 无 CPython 未定义符号**：
   - Linux：`nm -D --undefined-only libPTOASCompiler.so | grep -i 'Py\|_Python'` 应为空；恢复 `-z defs` 后链接必须通过（残留任何非 CPython 未定义符号都会在此暴露，须修而非重新放宽）。
   - macOS：`otool -L` / `nm -u` 不出现 CPython 符号。
3. **导入与构造等价**：装好后在**同一个 Python 版本**下：
   - `import ptoas._core` 成功；
   - `register_dialect` 正常；
   - `VRegType/MaskType/VMIVRegType/VMIMaskType/AlignType/StructType/TileBufType` 以及 PtrType/TileType 等的 `.get(...)`、各 getter、`isinstance` 判定与解耦前一致；
   - `StructType.get([])`（空 fields）与非法 field 类型走到清晰的 `value_error`。
4. **端到端**：跑一条既有回归 / `test/samples` 编译用例，确认编译期 C++↔Python 回调（TileLib/SoftLib）通过、产物一致。
5. **回归对照**：解耦前后同一用例的 IR/产物 diff 为空——证明解耦行为等价。

## 交叉层同步检查

本次改动落在「C API + Python 绑定 + 构建」三层，须同步：

- [ ] `include/pto-c/Dialect/PTOCAPI.h`：早期块补 `MLIR_CAPI_EXPORTED`；新增 35 声明。
- [ ] `lib/CAPI/Dialect/PTO.cpp`：新增 35 实现，上游 MLIR C++ 调用全封在此。
- [ ] `lib/Bindings/Python/PTOModule.cpp`：7 处类型改走 C API，不再触碰上游 C++ 符号。
- [ ] `tools/ptoas/CMakeLists.txt`：源文件迁移；DSO 去 pybind11、去 `-z undefs`、去 per-source rtti/exceptions；保留 hidden 可见性。
- [ ] 新增/改动文件按 `.claude/CLAUDE.md` 要求带 OAT.3 许可头。
- [ ] 单版本回归等价（导入、类型构造、编译期回调、端到端用例全通过）。

## 不做的事（本流程边界）

- 不改多版本构建流程（`build.sh` 的 N 解释器编排、`.ptoas-python-version` 集合、重定位脚本登记 N 份 `_core`、wrapper 集合校验）——那是解耦落地后的独立后续。
- 不改发布模型（仍沿用当前打包路径），仅确保解耦本身行为等价、DSO 版本无关。
