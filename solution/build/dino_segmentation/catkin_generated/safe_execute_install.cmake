execute_process(COMMAND "/code/solution/build/dino_segmentation/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/code/solution/build/dino_segmentation/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
