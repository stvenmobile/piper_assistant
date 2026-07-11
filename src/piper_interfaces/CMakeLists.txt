cmake_minimum_required(VERSION 3.8)
project(piper_interfaces)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(COMPILER_WARNINGS)
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find native build tools and action dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(action_msgs REQUIRED) # 💡 MANDATORY FOR ROS 2 ACTIONS

# Define your interface assets explicitly
set(action_files
  "action/ExecuteResearch.action"
)

# Generate typesupport matrices with action dependencies declared
rosidl_generate_interfaces(${PROJECT_NAME}
  ${action_files}
  DEPENDENCIES action_msgs
)

ament_package()