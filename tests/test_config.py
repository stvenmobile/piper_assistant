from piper_brain.config import DEFAULTS, deep_merge, load_config


def test_deep_merge_keeps_unmentioned_keys():
    merged = deep_merge({"a": {"x": 1, "y": 2}, "b": 3}, {"a": {"y": 20}})
    assert merged == {"a": {"x": 1, "y": 20}, "b": 3}


def test_deep_merge_does_not_modify_inputs():
    base = {"a": {"x": 1}}
    deep_merge(base, {"a": {"x": 2}})
    assert base == {"a": {"x": 1}}


def test_missing_file_gives_defaults(tmp_path):
    cfg = load_config(tmp_path / "nope.yaml", env={})
    assert cfg == DEFAULTS


def test_partial_section_keeps_other_defaults(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("audio:\n  tts_engine: kokoro\n", encoding="utf-8")
    cfg = load_config(path, env={})
    assert cfg["audio"]["tts_engine"] == "kokoro"
    assert cfg["audio"]["hardware_rate"] == DEFAULTS["audio"]["hardware_rate"]
    assert cfg["assistant"]["engaged_timeout_seconds"] == DEFAULTS["assistant"]["engaged_timeout_seconds"]


def test_environment_overrides_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("llm:\n  model: from-file\n", encoding="utf-8")
    cfg = load_config(path, env={"PIPER_LLM_MODEL": "from-env"})
    assert cfg["llm"]["model"] == "from-env"
    assert cfg["llm"]["base_url"] == DEFAULTS["llm"]["base_url"]


def test_repo_config_loads():
    from piper_brain.config import CONFIG
    for section in ("assistant", "llm", "audio", "weather"):
        assert section in CONFIG
