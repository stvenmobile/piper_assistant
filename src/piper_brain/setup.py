from setuptools import find_packages, setup
import os

package_name = 'piper_brain'

def get_flat_data_files(source_dir, target_share_path):
    """
    Walks the source directory and strips prefixes so files copy 
    directly into the root of the target share path destination.
    """
    data_files_map = []
    for root, dirs, filenames in os.walk(source_dir):
        for filename in filenames:
            source_file_path = os.path.join(root, filename)
            
            relative_path = os.path.relpath(root, source_dir)
            if relative_path == ".":
                destination_dir = target_share_path
            else:
                destination_dir = os.path.join(target_share_path, relative_path)
                
            data_files_map.append((destination_dir, [source_file_path]))
    return data_files_map

# Build direct target destination mappings
data_files_list = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
]

# 💡 Fixed: 'launch' is at the root of the package directory, not nested inside the module!
data_files_list.extend(get_flat_data_files('launch', 'share/' + package_name + '/launch'))

# Dynamically append flattened front-end and log/task infrastructure files
data_files_list.extend(get_flat_data_files('piper_brain/templates', 'share/' + package_name + '/templates'))
data_files_list.extend(get_flat_data_files('../piper_tools/piper_tools/assets', 'share/' + package_name + '/assets'))
data_files_list.extend(get_flat_data_files('../piper_tools/piper_tools/assets/world_model_vault', 'share/' + package_name + '/tasks'))

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=data_files_list,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='steve',
    maintainer_email='steve@todo.todo',
    description='Decoupled Object Telemetry Dashboard Core for Piper Assistant Stack',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'piper_supervisor = piper_brain.piper_supervisor:main',
            'dashboard_node = piper_brain.dashboard_node:main',
            'autonomous_drawing = piper_brain.autonomous_drawing:main',
        ],
    },
)