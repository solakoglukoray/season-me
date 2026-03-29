"""Color theory: season palettes and classification logic."""

SEASON_PALETTES: dict[str, list[dict]] = {
    "Spring": [
        {"name": "Warm Coral", "hex": "#FF6B6B"},
        {"name": "Peach", "hex": "#FFAB76"},
        {"name": "Ivory", "hex": "#FFFFF0"},
        {"name": "Golden Yellow", "hex": "#FFD700"},
        {"name": "Clear Mint", "hex": "#98FF98"},
        {"name": "Warm Aqua", "hex": "#7FFFD4"},
        {"name": "Camel", "hex": "#C19A6B"},
        {"name": "Salmon", "hex": "#FA8072"},
    ],
    "Summer": [
        {"name": "Lavender", "hex": "#E6E6FA"},
        {"name": "Dusty Rose", "hex": "#DCAE96"},
        {"name": "Powder Blue", "hex": "#B0C4DE"},
        {"name": "Mauve", "hex": "#E0B0FF"},
        {"name": "Soft Plum", "hex": "#8E4585"},
        {"name": "Blue Gray", "hex": "#6699CC"},
        {"name": "Rose Beige", "hex": "#E8C5A8"},
        {"name": "Soft Fuchsia", "hex": "#C74375"},
    ],
    "Autumn": [
        {"name": "Terracotta", "hex": "#E2725B"},
        {"name": "Rust", "hex": "#B7410E"},
        {"name": "Olive", "hex": "#808000"},
        {"name": "Deep Teal", "hex": "#008080"},
        {"name": "Burnt Orange", "hex": "#CC5500"},
        {"name": "Warm Brown", "hex": "#964B00"},
        {"name": "Moss Green", "hex": "#8A9A5B"},
        {"name": "Caramel", "hex": "#C68642"},
    ],
    "Winter": [
        {"name": "True Black", "hex": "#000000"},
        {"name": "Pure White", "hex": "#FFFFFF"},
        {"name": "Navy Blue", "hex": "#000080"},
        {"name": "Crimson", "hex": "#DC143C"},
        {"name": "Royal Purple", "hex": "#7851A9"},
        {"name": "Icy Pink", "hex": "#FFB3DE"},
        {"name": "True Blue", "hex": "#0000FF"},
        {"name": "Hot Pink", "hex": "#FF69B4"},
    ],
}

SEASON_DESCRIPTIONS: dict[str, str] = {
    "Spring": (
        "Warm, light, and clear — your natural coloring has golden undertones "
        "and a fresh, bright quality. You shine in warm, clear colors that echo "
        "the freshness of spring."
    ),
    "Summer": (
        "Cool, light, and muted — your natural coloring has pink or blue "
        "undertones with a soft, blended quality. You look best in dusty, muted "
        "tones that complement your delicate clarity."
    ),
    "Autumn": (
        "Warm, deep, and muted — your natural coloring has golden or earthy "
        "undertones with rich, complex tones. Rich, earthy, and muted shades "
        "that echo autumn foliage suit you perfectly."
    ),
    "Winter": (
        "Cool, deep, and clear — your natural coloring has pink or blue "
        "undertones with high contrast. Bold, clear, and icy colors that create "
        "sharp contrast are your power palette."
    ),
}


def classify_season(hue: float, saturation: float, lightness: float) -> str:
    """Classify personal color season from skin tone HSL values.

    Args:
        hue: Skin tone hue in degrees (0–360).
        saturation: Skin tone saturation (0.0–1.0).
        lightness: Skin tone lightness (0.0–1.0).

    Returns:
        One of "Spring", "Summer", "Autumn", "Winter".
    """
    # Undertone: warm (yellow-orange) vs cool (pink-red)
    # Warm skin hues cluster around 20–55° (yellow-orange tones)
    # Cool skin hues sit closer to 0–15° or >340° (pink-red tones)
    is_warm = 15.0 <= hue <= 55.0

    # Depth: light (L > 0.50) vs deep
    is_light = lightness > 0.50

    if is_warm and is_light:
        return "Spring"
    if not is_warm and is_light:
        return "Summer"
    if is_warm and not is_light:
        return "Autumn"
    # cool + deep
    return "Winter"
