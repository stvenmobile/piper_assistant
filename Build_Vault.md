Run these exact commands from your main workspace directory (~/piper_assistant) to build the Obsidian vault:
```Bash

cd /home/steve/piper_assistant

rm src/piper_tools/piper_tools/assets/model_concepts.md

# 1. Run the cloud reasoning pass to populate tasks/model_concepts.md
python3 -m piper_tools.run_synthesis

# 2. Explode definitions into individual Obsidian concept nodes
python3 -m piper_tools.obsidian_vault_builder

# 3. Export your database records into the Sources directory
python3 -m piper_tools.vault_data_linker

# 4. Generate the continuous reading dashboards
python3 -m piper_tools.build_reading_dashboards

```


