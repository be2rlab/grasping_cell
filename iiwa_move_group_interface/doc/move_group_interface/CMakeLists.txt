add_executable(move_group_interface_iiwa src/move_group_interface_iiwa.cpp)
add_executable(client src/client.cpp)
target_link_libraries(move_group_interface_iiwa ${catkin_LIBRARIES} ${Boost_LIBRARIES})
target_link_libraries(client ${catkin_LIBRARIES} ${Boost_LIBRARIES})


install(TARGETS move_group_interface_iiwa DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
