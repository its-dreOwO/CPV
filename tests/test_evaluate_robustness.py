from pathlib import Path

from scripts.evaluate_robustness import group_image_names


def _write_csv(tmp_path: Path) -> Path:
    csv = tmp_path / "attributes.csv"
    csv.write_text(
        "name,split,weather,scene,timeofday\n"
        "a.jpg,test,clear,city street,daytime\n"
        "b.jpg,test,rainy,highway,night\n"
        "c.jpg,test,clear,city street,night\n"
        "d.jpg,train,clear,city street,daytime\n"
    )
    return csv


def test_group_by_timeofday_filters_to_split(tmp_path):
    groups = group_image_names(_write_csv(tmp_path), split="test", by="timeofday")
    assert groups["daytime"] == ["a.jpg"]
    assert sorted(groups["night"]) == ["b.jpg", "c.jpg"]
    assert "d.jpg" not in groups.get("daytime", [])  # train split excluded


def test_group_by_weather(tmp_path):
    groups = group_image_names(_write_csv(tmp_path), split="test", by="weather")
    assert sorted(groups["clear"]) == ["a.jpg", "c.jpg"]
    assert groups["rainy"] == ["b.jpg"]
