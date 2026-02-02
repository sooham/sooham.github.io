# Typography Style Guide - Basquiat Aesthetic

This website uses a **Basquiat-inspired** typographic system that emphasizes bold, impactful, street art aesthetics while maintaining high legibility and readability.

## Philosophy

Jean-Michel Basquiat's work is characterized by:
- **Raw, expressive text** - Bold and unfiltered communication
- **High contrast** - Clear visual hierarchy
- **Street art influence** - Urban, graffiti-like elements
- **Geometric forms** - Clean, architectural shapes
- **Accessible expression** - Art that communicates directly

Our typography follows these principles with modern, web-optimized fonts that are:
- Highly legible across all devices
- Bold and impactful without sacrificing readability
- Clean and geometric while maintaining personality
- Optimized for both display and body text

## Font Stack

### Primary Fonts

1. **Bebas Neue** - Display/Heading Font
   - Usage: Hero names, section headings, navigation, card titles
   - Characteristics: Bold, condensed, all-caps feel, high impact
   - Fallback: 'Impact', sans-serif
   - CSS Variable: `--font-hand`

2. **Work Sans** - Body/Text Font
   - Usage: Paragraphs, descriptions, subtitles, UI text
   - Characteristics: Geometric, clean, highly readable, versatile weights
   - Weights used: 400 (regular), 500 (medium), 600 (semi-bold), 700 (bold), 800 (extra-bold), 900 (black)
   - Fallback: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif
   - CSS Variable: `--font-sketch`

3. **Space Mono** - Monospace/Code Font
   - Usage: Code blocks, technical tags, skill badges
   - Characteristics: Geometric monospace, technical feel
   - Weights used: 400 (regular), 700 (bold)
   - Fallback: 'Courier New', monospace
   - CSS Variable: `--font-mono-hand`

## Typography Hierarchy

### Display Typography (Bebas Neue)

#### Hero Name
```stylus
font-family: 'Bebas Neue'
font-size: 3.8rem (desktop), 2.8rem (mobile)
font-weight: 400
letter-spacing: 2px (desktop), 1px (mobile)
text-transform: uppercase
```

#### Section Headings
```stylus
font-family: 'Bebas Neue'
font-size: 2.2rem (desktop), 1.8rem (mobile)
font-weight: 400
letter-spacing: 3px (desktop), 2px (mobile)
text-transform: uppercase
```

#### Navigation Links
```stylus
font-family: 'Bebas Neue'
font-size: 1.4rem
font-weight: 400
letter-spacing: 2px
text-transform: uppercase
```

#### Card Titles
```stylus
font-family: 'Bebas Neue'
font-size: 1.7rem - 2rem
font-weight: 400
letter-spacing: 1px
text-transform: uppercase
```

#### Company Names
```stylus
font-family: 'Bebas Neue'
font-size: 1.5rem (desktop), 1.3rem (mobile)
font-weight: 400
letter-spacing: 1px
text-transform: uppercase
```

### Body Typography (Work Sans)

#### Hero Subtitle & Title
```stylus
font-family: 'Work Sans'
font-size: 1.1rem - 1.3rem
font-weight: 600 (title), 500 (subtitle)
letter-spacing: 0.3px - 0.5px
```

#### Paragraphs & Descriptions
```stylus
font-family: 'Work Sans'
font-size: 1rem - 1.15rem
font-weight: 500
line-height: 1.6 - 1.9
```

#### Role & Metadata
```stylus
font-family: 'Work Sans'
font-size: 0.95rem - 1rem
font-weight: 500
```

#### Menu Links
```stylus
font-family: 'Work Sans'
font-weight: 600
```

### Monospace Typography (Space Mono)

#### Code Blocks
```stylus
font-family: 'Space Mono'
font-size: 85% of parent
font-weight: 400
line-height: 1.8
```

#### Tech Tags
```stylus
font-family: 'Space Mono'
font-size: 0.85rem - 0.9rem
font-weight: 400
```

## Color Palette

### Text Colors (from CSS Custom Properties)
- `--text-ink: #2c2824` - Primary dark text (headings, emphasis)
- `--text-pencil: #3d3832` - Main body text
- `--text-light: #6b6359` - Secondary text (metadata, captions)

### Accent Colors
- `--accent-terracotta: #c45d3a` - Primary accent (links, highlights)
- `--accent-sage: #6b8f71` - Secondary accent (active states, badges)
- `--accent-mustard: #d4a84b` - Tertiary accent (decorative elements)

### Background Colors
- `--bg-cream: #faf8f3` - Primary background
- `--bg-paper: #f7f4ed` - Secondary background (cards, sections)

## Font Loading

Fonts are loaded via Google Fonts in `/themes/polk/layout/_partial/head.ejs`:

```html
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Work+Sans:wght@400;500;600;700;800;900&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">
```

## Usage Guidelines

### DO:
- Use **Bebas Neue** for all headings and display text that needs impact
- Always apply `text-transform: uppercase` with Bebas Neue
- Use appropriate letter-spacing (1-3px) with Bebas Neue to improve readability
- Use **Work Sans** (weights 500-700) for all body text and descriptions
- Use **Space Mono** for all code, technical content, and developer-focused UI elements
- Maintain consistent font weights across similar elements
- Use high contrast between text and background

### DON'T:
- Don't use Bebas Neue for long paragraphs (use Work Sans instead)
- Don't use font-weight with Bebas Neue (it only has one weight, use size and spacing instead)
- Don't use font-style: italic with Work Sans in hero sections (use font-weight variation)
- Don't mix cursive or handwritten fonts with this system
- Don't use Comic Sans MS or similar informal fonts
- Don't use font sizes smaller than 0.85rem for body text

## Responsive Typography

### Mobile Breakpoints

#### @media (max-width: 600px)
- Hero name: 2.8rem → 2.4rem
- Section headings: 2.2rem → 1.8rem
- Reduce letter-spacing proportionally

#### @media (max-width: 480px)
- Page titles: Further reduce to 2rem - 2.2rem
- Maintain letter-spacing at minimum 1px for legibility

## Accessibility

- **Font smoothing**: `-webkit-font-smoothing: antialiased` and `-moz-osx-font-smoothing: grayscale` applied globally
- **Line height**: Minimum 1.6 for body text, 1.2 for headings
- **Contrast**: All text meets WCAG AA standards (4.5:1 for body, 3:1 for large text)
- **Font size**: Base font size is 16px (1rem), never below 14px (0.875rem)

## File Structure

### Font Declarations
- `/themes/polk/layout/_partial/head.ejs` - Font imports
- `/themes/polk/source/css/style.styl` - Base font stack
- `/themes/polk/source/css/_partial/landing.styl` - CSS custom properties and landing page typography
- `/themes/polk/source/css/_partial/visualizations.styl` - Visualization page typography
- `/themes/polk/source/css/_partial/post.styl` - Blog post typography
- `/themes/polk/source/css/_partial/header.styl` - Header/navigation typography
- `/themes/polk/source/css/_partial/archive.styl` - Archive page typography

## Version History

### v2.0 - Basquiat Aesthetic (Current)
- Replaced Caveat with Bebas Neue for display typography
- Replaced Architects Daughter with Work Sans for body typography
- Replaced Indie Flower with Space Mono for monospace
- Improved legibility and readability across all devices
- Enhanced visual hierarchy with bold, geometric fonts
- Maintained street art aesthetic with cleaner execution

### v1.0 - Hand-Drawn Aesthetic (Previous)
- Used Caveat for headings
- Used Architects Daughter for body
- Used Indie Flower for monospace
- Issues: Lower legibility, harder to read on small screens

## Design Principles

1. **Hierarchy First**: Clear visual distinction between heading levels
2. **Readability Always**: Never sacrifice legibility for style
3. **Consistency**: Same elements use same typography
4. **Impact with Purpose**: Bold typography serves the content
5. **Basquiat Spirit**: Raw, direct, powerful communication
