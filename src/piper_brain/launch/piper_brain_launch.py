import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 🛡️ 1. Core State Supervisor Node (piper_brain Package)
        Node(
            package='piper_brain',
            executable='piper_supervisor',
            name='piper_supervisor',
            output='screen',
            emulate_tty=True, # Ensures colorized logging output prints beautifully in the terminal
            parameters=[
                # Future configuration parameters can be added here cleanly
            ]
        ),
        
        # 🎓 2. Hermes Research Action Server Node (piper_tools Package)
        Node(
            package='piper_tools',
            executable='research_node',
            name='hermes_research_node',
            output='screen',
            emulate_tty=True,
            parameters=[
                # Future tool parameters can be placed here
            ]
        ),

        # 💡 Dashboard Node Execution Alignment
        Node(
            package='piper_brain',
            executable='dashboard_node',
            name='um790_dashboard_node',
            output='screen'
        ),
        
        # The Autonomous Drawing Node for quick sketches
        Node(
            package='piper_brain',
            executable='autonomous_drawing', # Assumes 'autonomous_drawing' is set in your piper_brain setup.py
            name='autonomous_drawing',
            output='screen'
        ),
    ])
