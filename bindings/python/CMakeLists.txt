set_target_properties(
  gtn
  PROPERTIES
  INTERFACE_POSITION_INDEPENDENT_CODE
  ON
  )

function (add_pybind11_extension ext_name)
  pybind11_add_module(
    ${ext_name}
    ${PROJECT_SOURCE_DIR}/bindings/python/gtn/${ext_name}.cpp
    )

endfunction()

add_pybind11_extension(_graph) 
add_pybind11_extension(_functions)
add_pybind11_extension(_utils) 
add_pybind11_extension(_autograd)
add_pybind11_extension(_torch)

