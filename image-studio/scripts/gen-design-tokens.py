#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TOKENS_PATH = ROOT / "tokens" / "design-tokens.json"
SWIFT_OUT = ROOT / "apple" / "Bonsai" / "Generated" / "DesignTokens.swift"
TAILWIND_OUT = ROOT / "tokens" / "generated" / "tailwind.tokens.cjs"


def swift_identifier(name: str) -> str:
    value = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    if not value:
        raise ValueError(f"Cannot convert token name {name!r} to a Swift identifier.")
    if value[0].isdigit():
        value = f"_{value}"
    return value


def parse_color(value: str) -> tuple[float, float, float, float]:
    value = value.strip()
    if value.startswith("#"):
        hex_value = value[1:]
        if len(hex_value) == 6:
            red = int(hex_value[0:2], 16)
            green = int(hex_value[2:4], 16)
            blue = int(hex_value[4:6], 16)
            return (red / 255.0, green / 255.0, blue / 255.0, 1.0)
        raise ValueError(f"Unsupported hex color {value!r}.")
    match = re.fullmatch(
        r"rgba\(\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)",
        value,
    )
    if match is None:
        raise ValueError(f"Unsupported color format {value!r}.")
    red, green, blue, alpha = match.groups()
    return (int(red) / 255.0, int(green) / 255.0, int(blue) / 255.0, float(alpha))


def color_line(name: str, raw_value: str) -> str:
    red, green, blue, alpha = parse_color(raw_value)
    return (
        f"        static let {swift_identifier(name)} = BonsaiColorToken("
        f"rawValue: {json.dumps(raw_value)}, red: {red:.6f}, green: {green:.6f}, "
        f"blue: {blue:.6f}, alpha: {alpha:.6f})"
    )


def numeric_line(name: str, value: int | float) -> str:
    literal = f"{value:.3f}" if isinstance(value, float) else str(value)
    return f"        static let {swift_identifier(name)}: CGFloat = {literal}"


def generate_swift(tokens: dict) -> str:
    colors = "\n".join(color_line(name, token["value"]) for name, token in tokens["color"].items())
    spacing = "\n".join(numeric_line(name, value) for name, value in tokens["spacing"].items())
    radii = "\n".join(numeric_line(name, value) for name, value in tokens["radius"].items())
    font_sizes = "\n".join(numeric_line(name, value) for name, value in tokens["typography"]["sizes"].items())
    line_heights = "\n".join(
        numeric_line(name, value) for name, value in tokens["typography"]["lineHeights"].items()
    )
    tracking = "\n".join(numeric_line(name, value) for name, value in tokens["typography"]["tracking"].items())

    return f"""import SwiftUI

struct BonsaiColorToken: Sendable, Equatable {{
    let rawValue: String
    let red: Double
    let green: Double
    let blue: Double
    let alpha: Double

    var color: Color {{
        Color(.sRGB, red: red, green: green, blue: blue, opacity: alpha)
    }}
}}

enum DesignTokens {{
    enum Colors {{
{colors}
    }}

    enum Spacing {{
{spacing}
    }}

    enum Radius {{
{radii}
    }}

    enum Typography {{
        static let sans = {json.dumps(tokens["typography"]["fontFamilies"]["sans"])}
        static let mono = {json.dumps(tokens["typography"]["fontFamilies"]["mono"])}

        enum Sizes {{
{font_sizes}
        }}

        enum LineHeights {{
{line_heights}
        }}

        enum Tracking {{
{tracking}
        }}
    }}
}}
"""


def generate_tailwind(tokens: dict) -> str:
    payload = {
        "colors": {name: token["value"] for name, token in tokens["color"].items()},
        "spacing": tokens["spacing"],
        "borderRadius": tokens["radius"],
        "fontFamily": tokens["typography"]["fontFamilies"],
        "fontSize": tokens["typography"]["sizes"],
        "lineHeight": tokens["typography"]["lineHeights"],
        "tracking": tokens["typography"]["tracking"],
    }
    json_blob = json.dumps(payload, indent=2, sort_keys=True)
    return f"module.exports = {json_blob};\n"


def main() -> None:
    tokens = json.loads(TOKENS_PATH.read_text())
    SWIFT_OUT.write_text(generate_swift(tokens))
    TAILWIND_OUT.write_text(generate_tailwind(tokens))
    print(f"Wrote {SWIFT_OUT.relative_to(ROOT)}")
    print(f"Wrote {TAILWIND_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
