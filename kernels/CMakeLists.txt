# Kernels are executables rather than tests: they render, and correctness is
# judged by looking at the image.
add_executable(ray_triangle ray_triangle.cpp)
target_link_libraries(ray_triangle PRIVATE gpurt project_warnings)

# The same scene through the rasteriser. Both write a PPM, and the two should
# agree — which is the check neither one can make on its own.
add_executable(raster_triangle raster_triangle.cpp)
target_link_libraries(raster_triangle PRIVATE gpurt project_warnings)

# A mesh through every route, with an image and a number from each. The two
# above hold one hard-coded triangle, which is what makes them a fixed reference
# between the renderers and also what stops them showing a shared vertex.
add_executable(mesh_render mesh_render.cpp)
target_link_libraries(mesh_render PRIVATE gpurt project_warnings)

# Written relative to the working directory, so run it from the project root.
# The directory is tracked via benchmarks/result/.gitkeep; create it if a fresh clone has
# not got one yet.
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)
