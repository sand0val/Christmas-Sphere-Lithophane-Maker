# Lithophane Sphere Studio

Turns 1 to 4 photos into a lithophane Christmas bauble for the **Bambu Lab A1** in **white PLA**,
with a greeting and Christmas decorations on the pictures - and on the rest of the ball - if you
like. The ornament prints in one
piece with no supports, hangs from a ribbon or hook, and glows when a tree light is pushed into
its cap - or a battery light is fed in, or sits in a seat on top.

## Start it

Double-click **`Lithophane Sphere Studio.bat`**, or run:

```
python main.py
```

First run only: `python -m pip install -r requirements.txt`

## Using it

The left side is six steps, in order. The right side is a live 3D preview.

| Step | What you set |
|------|--------------|
| **1 Pictures** | How many pictures (1-4) and which photos. Click a picture to zoom, move, rotate, mirror or brighten it. Drag photos straight from Explorer. |
| **2 Decorate** | A greeting and Christmas decorations for each picture: lettering over the photo, garlands, fairy lights, holly, snow, a wreath, a candy-cane border, snowflakes. A picture without a photo becomes a panel of lettering only. At the foot of the page: decorations for the plain parts of the ball itself (see "Lettering and decorations" below). |
| **3 Sphere** | Diameter, how tall the pictures are, the gap between them, how a flat photo is fitted to the ball, and which side of the wall carries the picture (see "Which side carries the picture" below). |
| **4 Light & dark** | Thinnest / thickest wall, frame thickness, mid-tone brightness, smoothing, print-safe edges. |
| **5 Cap** | Which light you use (sets the opening, or a seat for a light that sits on the cap), cap size, shoulder angle, hanger loops. |
| **6 Print** | Nozzle (0.4 or 0.2 mm), mesh detail, a live checklist, and the exact Bambu Studio settings. |

**Rest the mouse on anything** - a slider, a button, a tab, an entry of a list, a line of the
checklist, a menu entry - and a box says what it does and how to choose. Sliders also say where
they start. `Help > Show tips` switches the boxes off once you know your way round.

Preview modes (top of the 3D view, or F1-F4):

- **Glowing** - a simulation of light through white PLA: this is what it looks like on the tree.
- **Unlit** - the plain white print.
- **Cutaway** - sliced in half so you can see the wall thickness and the inside.
- **On the printer** - upside-down on the bed, exactly as exported.

Drag to turn, scroll to zoom, Shift-drag to move (View > Reset view, or Home, puts it back).
Double-click a slider's name to restore its default. Linked settings adjust each other so ordinary
edits cannot produce an unprintable shape; if something truly cannot work, a red banner says what
to change.

The 3D view is drawn off-screen and shown as an ordinary picture. That is deliberate: overlay
software that hooks into 3D windows cannot freeze it. (The Nahimic audio overlay, which Bambu
Studio also warns about when it starts, hangs normal OpenGL windows on a second monitor on some
PCs - this app is not affected, on any monitor.)

`File > Save project` keeps the photos' locations and every setting in a small `.lsphere` file.

## Lettering and decorations

On the **Decorate** page, choose a picture and:

- **Type a greeting.** Enter starts a new line (up to six). The block is fitted into the picture,
  so the longest line decides how large everything can be - "Happy Holidays from Johnny and
  Rachel 2026" on one line comes out tiny, on four lines it is easy to read. **Size** is a share
  of the largest size that fits; the line under the box gives the letter height in millimetres.
- **Pick a font.** Every font on the computer is listed, sturdy festive ones first (Cooper Black,
  Harlow, Forte, the Segoe scripts, Comic Sans ...). A font installed in Windows appears the next
  time the app starts. **Thicken letters** makes a fine font sturdy enough to print.
- **Glowing or dark letters.** Glowing letters are the thinnest wall, with a dark edge round them
  so that they read on any photo; dark letters are the other way round.
- **Decorate.** Four lists, one for each place: *along the top* (fir garland with baubles and
  bows, string of fairy lights, snow and icicles, holly in the corners), *along the bottom*
  (garland, bank of snow, snow with fir trees, holly), *all the way round* (line frame,
  candy-cane border, round wreath, wreath along the edges) and *scattered* (snowflakes, falling
  snow, stars, hearts - **Shuffle** deals them afresh). Lettering keeps out of the strip a
  garland takes up, and scattered shapes keep off the lettering, the other decorations and the
  middle of a photo. "Use these decorations on every picture" makes the pictures match.
- **Borders run right along the edge of the picture.** What a border covers is hidden and no
  sliver of the photo is left outside it. The **wreath** is fir - a dark bough thick with glowing
  needles, baubles all round, a bow at the foot - and comes in two shapes: *round*, which turns
  the picture into a round window with plain corners (lettering then goes inside the ring), and
  *along the edges*, which covers only the border of the photo. The **candy-cane border** is a
  broad glowing band with bold dark stripes on the slant, one wide and one narrow in turn. On
  narrow pictures (four on a ball) or low ones, a wreath or candy-cane border is drawn smaller
  than **Size of the decorations** asks, so that it never swallows the picture.
- **A picture of lettering only.** Remove a picture's photo (the cross on the Pictures page) and
  it becomes a panel that carries just its lettering and decorations: plain like the frame, dark,
  or bright. Three photos and a fourth picture with the greeting is a good use of four pictures.

**The rest of the ball.** The plain parts of the ornament can be decorated too, with the four
lists at the foot of the Decorate page:

| Where | What | How it comes out |
|-------|------|------------------|
| **Between the pictures** | stars, snowflakes, a string of beads, candy-cane stripes, a fir branch - up every divider | Crisp below the middle of the ball, softer above it, like a picture. The divider at the back stays plain: the seam runs down it. |
| **Round the top** | a row of stars, snowflakes or beads, a zigzag, a fir garland | Formed on the inside of the wall, so it shows softly when the ornament is lit and not at all unlit. (Up there the outside faces down on the printer and cannot carry a pattern.) |
| **Round the bottom** | the same choices | Cut into the outside: crisp, and it shows lit and unlit. |
| **Underneath** | a big snowflake or a big star on the closed end | Stands 0.6 mm proud of the ball instead of being cut in - the wall down there closes the ball and is never made thinner. Lit, it is a darker shape on the glowing bottom. |

(That is with the pictures on the outside of the wall, the usual setting. The ball's decorations
are made the way the pictures are: with **Inside** they are formed on the inside too - soft, and
visible only when lit - and there is no ring round the bottom, because that strip is then where
the inside of the wall runs in to the closed end.)

These shapes are plain glowing ones, without the dark edge that shapes on a picture have: against
plain frame they need none, and without it the printer has far fewer tiny separate outlines to
lay down. A ring leaves a gap of a few millimetres at the back, where the seam crosses it. Each
decoration needs room: with the default 12 degree gap and 100 degree picture height everything
fits except the fir garland round the top or bottom, which needs pictures of 97 degrees or less.
When something does not fit it is left out, and the checklist says which setting to change and
to what.

Lettering and decorations are part of the picture: light is the thinnest wall, dark the thickest.
Every shape is drawn where it will be on the ball, so a round light is round and a 1 mm stroke
1 mm wide wherever it sits, and they are drawn in the finished wall's own tones - the Smoothing,
Mid-tone and Negative settings change the photo underneath, never the lettering.

**Where it is sharp, and where it is not.** With the picture on the outside of the wall (the
default), the half of every picture nearest the cap is printed with its outside facing down, and
there the printer cannot build a sudden edge (see "Which side carries the picture"). A photo
hardly notices: it has few such edges. Lettering and decorations are nothing but sudden edges.
So:

- **Below the middle of a picture everything is crisp.** That is where lettering goes unless you
  move it, and the place for anything fine.
- **Above the middle it gets softer the higher it sits.** A light shape keeps a crisp upper edge
  and gets a soft lower one (a dark shape the other way round), which reads as a gentle embossed
  look. Shapes a few millimetres across stay readable right to the top - which is why the
  decorations for the top edge are drawn bold - but small lettering up there smears. The Glowing
  view shows what to expect, and the checklist warns when lettering is too fine for where it
  sits.
- With **Both** or **Inside** the upper half (or all) of a picture is shaped on the inside of the
  wall and is soft throughout, lettering included.

**How small can letters be?** With the 0.4 mm nozzle, letters want to be about 3 mm tall or more
with strokes of 0.7 mm; with the 0.2 mm nozzle about half that. Below it the gaps inside the
letters close up. The checklist says so, and says what to do: bigger, shorter lines, a sturdier
font or Thicken letters. (Those limits come from what the slicer makes of grooves that narrow, not
from a print: nothing with lettering has been printed here yet.) If a font has no shape for a
character - a heart or a star in a font of plain letters - the checklist says that too; Segoe UI
Symbol has them.

## Printing

**Export for printing...** writes three files:

| File | What it is |
|------|------------|
| `name_bambu_A1.3mf` | **Ready-to-print Bambu Studio project.** The ornament sits cap-down in the middle of the plate with the A1, textured plate, PLA and every process setting already chosen, and with the seam line marked (see "The seam" below). Open it, choose your white filament, press **Slice plate**, then **Print plate**. |
| `name.stl` | The plain model, already turned the right way up, for any slicer. An STL cannot carry the seam line; choose "3MF model" in the save box instead and the plain model carries it too. |
| `name_print_settings.txt` | The settings, if you slice the plain model yourself. |

The project is made by letting Bambu Studio slice the ornament in the background (under a
minute; a minute or two for the 0.2 mm nozzle), which also proves it slices cleanly and tells you
the print time and filament. It needs Bambu Studio installed; without it you still get the other
two files.

Keep supports **off**: no part of the ornament starts in mid-air, and supports inside the ball
could never be removed. Bambu Studio should slice it without a "floating regions" warning. If it
ever shows one, take it seriously - it means a picture detail hangs in mid-air and would print as
a blob. That can only happen with **Print-safe edges** switched off; the app warns about it on the
Print page and again when you export.

Use **LED** lights only: incandescent bulbs get hot enough to soften PLA. See "Lighting it" below
for battery lights.

### Which side carries the picture

A lithophane is a wall whose thickness follows the photo. Which face of the wall carries the
bumps matters twice: for how the picture looks, and for whether it can be printed.

**How it looks.** White PLA scatters light. Relief on the **outside** of the wall is the last
thing the light passes, so its edges stay sharp. Relief on the **inside** shapes the light first;
the light then spreads sideways in the plastic before it leaves through a smooth outside, dips on
the lit face sit partly in shadow, and thin dark details collect light through their sides. The
picture comes out blurred by about a millimetre: small bright dots turn into dim blobs and thin
dark lines all but vanish. The first ornament printed with this app (version 1.2, relief on
"Both" sides) showed exactly that - the half with the relief inside was soft, the half with it
outside was crisp - although the model and the sliced file were equally sharp in both halves. A
simulation of light scattering in the wall gives the same result, and other makers report it too
([cylinder](https://www.thingiverse.com/thing:322720),
[moon lamp](https://www.instructables.com/High-Detailed-Moon-Lamp/)).

**How it prints.** The ornament is printed upside-down, in one piece:

| Half of the ornament | Outside of the wall | Inside of the wall |
|----------------------|---------------------|--------------------|
| Nearest the cap (printed first; top half when it hangs) | faces down | faces up |
| Furthest from the cap (printed last; bottom half when it hangs) | faces up | faces down |

Relief on a face that points up can have any shape. On a face that points down, a detail that
thickens suddenly going up on the printer would hang lower than the place where it joins the
wall, so the printer reaches it before anything holds it: that is a floating region.

| Setting | What you get |
|---------|--------------|
| **Outside** (default) | The whole picture on the outside: sharp when lit, embossed when not. In the half printed last the outside faces up and carries everything. In the half printed first it faces down; it still carries every part of the picture where the wall gets thinner going up on the printer, or thicker gently. Only where it thickens too suddenly does the outside ramp instead - at 30 degrees from flat, the angle of the shoulder, centred on the edge - and the inside, which faces up there, takes a matching dip and bump (under a millimetre each in the demo), so the wall keeps the thickness the picture asks for. In the demo ornament about 5 % of the relief in the upper halves ends up inside. Lit, those few edges (bright above, dark below) are about a millimetre softer than the rest. |
| **Both** | Relief inside above the middle, outside below it. The upper half of the ornament is smooth, but the upper half of every picture is blurred when lit. This was the default of versions 1.1 and 1.2. |
| **Inside** | Smooth ornament all over; the whole picture is blurred when lit. In the lower half of each picture sharp bright-above-dark edges are also eased to 20 degrees from flat so that they can be printed. |

The **Glowing** preview shows the difference: it blurs whatever relief is on the inside of the
wall, the way a real print does. Projects and sessions saved with "Both" or "Inside" as the old
default open with **Outside**; a setting chosen from now on is kept.

**Never too thin.** Sharing an edge has one trap, which version 1.3 fell into. Where the inside
makes room for an outside that is still on its ramp, both faces move outwards together. Beside
the end of such an edge - next to a column that has no edge and stays where it was - the wall is
then a thin slanted web: as thick as the picture asks along the radius, but far thinner measured
square to its surface. In the version 1.3 demo the wall got down to about half a millimetre in a
few spots, and with a test photo of pure black and white blocks to 0.3 mm; the slicer joined the
outside and the inside wall loop there (5 of the demo's 511 picture layers). Since version 1.4
the inside gives way wherever the wall would be thinner than nine tenths of the thinnest wall
(0.72 mm by default) measured in space, and the wall is a little thicker than the picture asks
there instead. In the demo that changes 0.15 % of the picture area by a tenth of a millimetre on
average. "Both" and "Inside" never had the problem: one of their faces is always smooth.

The same limit explains the one thing the **Thinnest wall** setting cannot do with the 0.4 mm
nozzle: go much below 0.8 mm. The slicer lays two lines side by side - the outside and the
inside of the wall - only where the wall is at least 0.68 mm wide. With 0.8 mm every layer of
the demo is two full loops; with 0.6 mm, 195 of its 511 picture layers are not, and the
brightest areas print as a single line. The checklist warns below 0.76 mm. The 0.2 mm nozzle
needs half as much.

### The seam

Every layer has to start somewhere, and that start leaves a faint line up the print: the seam.
The ornament is turned so that a plain divider between two pictures faces the back of the
printer, and the seam is set to "Back" - but that alone does not put the seam on the divider.
Bambu Studio's Back seam is only roughly at the back: on a round object it drifts, by up to
about 14 degrees either side in the slices measured here, and where it settles changes from one
slice to the next. In one slice of the demo ornament half of the layers started in a picture,
up to 4 mm past the edge of a 12 degree divider; in another slice of the very same shape hardly
any did. A wider gap makes that less likely, but nothing short of about 30 degrees rules it out.

So the exported project marks a line, 2 degrees wide, down the middle of the back divider as
the place where layers must start. It is the same marking that Bambu Studio's own **Seam
painting** tool makes (an "enforcer"), and you can see it with that tool. With it, the seam of
every combination tried - one to four pictures, all three relief settings, 50 to 150 mm, both
nozzles, all three mesh qualities, gaps of 3, 6 and 12 degrees - runs down the divider, within
a millimetre of its centre line (but for the odd single loop described below).

- The gap between the pictures only has to be wide enough to hold that line: about 3 degrees
  on the 80 mm default, a little more on a smaller ball. Beyond that it is a matter of taste
  (12 degrees by default). The checklist on the Print page says so if the gap is too narrow.
- After slicing, the app reads the result and tells you where the layers really start - and
  whether every layer of the pictures (and of a ring of decoration above or below them) is a
  closed ring of two wall loops, the outside and the inside of the wall. (A layer that is not
  has a spot too thin for two lines side by side.)
- One thing the line cannot rule out: where steep relief ends in a ridge, a layer cuts the tip
  of the ridge off as a small island, and now and then Bambu Studio starts the wall loop next to
  such an island at the island instead of on the line (its matching of loops is not exact with
  the Arachne wall generator). That needs very sudden edges and is rare: 1 to 3 loops in 1,000
  to 2,000 with a test photo of pure black and white blocks, 2 in 1,022 with all four demo
  pictures (the fourth is lettering), none with the first three. Each such loop leaves a mark
  one layer tall. The app counts them as well and reports them for what they are.
- Another thing it now and then does with such ridges: it prints a wall loop in two passes -
  stops part of the way round, prints the ridge somewhere else, comes back and finishes. The
  wall is whole, but the second pass has a start of its own. The app puts such passes together
  again before it judges a layer, and counts the second starts that lie in a picture along with
  the loops above. (Seen once in 511 layers of an early, fully decorated test ornament, on a
  divider.)
- The lower the edge sits on the ball, the more such ridges it makes: the wall leans in further
  there. That is why the ball's own decorations are plain glowing shapes without the dark edge
  that shapes on a picture have. Sliced both ways, a ring of snowflakes round the bottom made 563
  tiny separate outlines in its 33 layers with edges and 278 without, and the two wall loops
  that had started at one of them started at the back again.
- **Fix model** and **Simplify** in Bambu Studio erase painted seams. Do not use them on the
  ornament (it needs neither).
- If you slice the plain STL yourself, draw the line with the Seam painting tool first.

### The settings (already inside the project)

| Setting | 0.4 mm nozzle | 0.2 mm nozzle |
|---------|---------------|---------------|
| Printer preset | Bambu Lab A1 0.4 nozzle | Bambu Lab A1 0.2 nozzle |
| Filament preset | Bambu PLA Basic @BBL A1 | Bambu PLA Basic @BBL A1 0.2 nozzle |
| Process preset | 0.12mm High Quality @BBL A1 | 0.06mm High Quality @BBL A1 0.2 nozzle |
| Layer height | 0.12 mm (0.08 mm preset if you lowered the shoulder angle below 30) | 0.06 mm |
| Wall loops | 6 | 11 |

Both presets print outer walls at 60 mm/s. The loops are counted so that the thickest wall is
walls only, with no infill pattern in the picture. Use white PLA (Bambu PLA Basic white works
well; matte and silk are too opaque). The rest is the same for both nozzles:

| Setting | Value |
|---------|-------|
| Sparse infill | 100 %, pattern Rectilinear (Bambu Studio refuses gyroid at 100 %) |
| Wall generator | Arachne |
| Seam position | Back, plus the seam line marked down the divider that faces the back |
| Supports | Off |
| Brim | Outer brim only, 5 mm |
| Orientation | As exported, cap on the bed. Do not rotate, auto-orient or auto-arrange: the divider with the seam line should stay at the back. |

### The 0.2 mm nozzle

**Nozzle** on the Print page switches the whole export to the 0.2 mm nozzle: Bambu Studio's own
presets for it (see the table above), wall loops counted for its 0.22 mm lines, and a model
sampled every 0.15 mm instead of 0.25 mm.

- **Why 0.06 mm layers.** The design's overhangs - shoulder, closed end, the ramps of the picture
  relief - are proven with 0.12 mm layers and 0.42 mm lines. Halving both keeps every one of them
  at the same share of a line per layer (47 % on the shoulder). Thicker layers with the thin
  lines would not: at 0.08 mm the hidden 20 degree ramps would step a whole line width sideways.
- **What it buys.** Measured on the sliced files of the demo, as the share of the wanted relief
  that is in the printed outline, by size of detail (one bright-and-dark cycle):

  | Detail | 0.4 mm nozzle | 0.2 mm nozzle |
  |--------|---------------|---------------|
  | 2 - 4 mm | 89 % | 91 % |
  | 1.4 - 2 mm | 67 % | 83 % |
  | 1 - 1.4 mm | 55 % | 80 % |
  | 0.7 - 1 mm | 33 % | 69 % |
  | 0.5 - 0.7 mm | 17 % | 57 % |

  That is what the printer is told to do. How much of it the eye sees through one to three
  millimetres of plastic that scatters light has not been measured here; makers who compared the
  two nozzles on lithophanes call the gain real but modest.
- **What it costs.** A line with a quarter of the cross-section needs four times the length:
  the demo takes 14 h 21 min instead of 4 h 45 min. Exporting takes a minute or two, and the
  files are about three times the size.
- **Before you print.** Fit the nozzle and change the nozzle size on the printer's screen - the
  A1 does not notice the swap. Use plain white PLA: Bambu rules out filaments with fibres,
  glitter, marble, wood or glow particles for this nozzle, and says clogs are more common with
  it anyway, so keep the filament clean and dry. The first layer is only 0.1 mm thick: wipe the
  plate first. Some makers found that so thin a first layer holds badly on the textured plate
  and well on a smooth one; nothing was printed with this nozzle here, so that is hearsay.

### What the slicer does with the demo ornament

Measured on the Bambu Studio slices of the built-in demo (80 mm, three pictures), which are in
`output/`:

| | 0.4 mm nozzle | 0.2 mm nozzle | 0.4 mm, seat for a puck |
|---|---|---|---|
| File | `christmas_sphere_demo_bambu_A1.3mf` | `christmas_sphere_demo_nozzle02_bambu_A1.3mf` | `christmas_sphere_demo_puck_bambu_A1.3mf` |
| Print time | 4 h 45 min | 14 h 21 min | 4 h 47 min |
| Filament | 48 g | 48 g | 50 g |
| Layers | 719 | 1,439 | 729 |
| Supports, "floating regions" warning | none | none | none |
| Filament that is plain walls | 97 % | 97 % | 98 % |
| Wall loops right round, all starting within 3 degrees of the back | 1,413 | 2,828 | 1,434 |
| Furthest start from the seam's centre line | 0.6 mm | 0.4 mm | 0.8 mm |
| Picture layers that are a closed ring of two loops | 511 of 511 | 1,022 of 1,022 | 511 of 511 |

- The app's own check finds no point that starts in mid-air in any of them.
- Each has exactly one bridged layer - the small ceiling inside the closed end, where the
  design puts it. None of the filament is flagged as overhanging wall.
- No downward-facing part of the outside is closer to flat than 30 degrees, the angle of the
  shoulder; inside, where it cannot be seen, the flattest is about 21 degrees.
- Eighteen more combinations were sliced the same way - one, two and four pictures, 50 to 150 mm,
  a taller picture band, thin and thick frames, gaps of 3 and 6 degrees, the finest mesh, the
  balloon-light opening, puck and tea-light seats, relief on Both sides and Inside - and a test
  photo made of nothing but pure black and pure white blocks (the hardest edges there are) in
  six more. No supports, no warning and a closed ring on every picture layer in all of them.
  The seam is on the divider in all of them too, apart from the odd single loop described under
  "The seam" (four pictures: 2 of 1,022 loops; the block photo: up to 3 of 2,044).

**With lettering and decorations.** Three ornaments made from the photos in `my_pics/` are in
`output/` as well, each with its project (`.lsphere`) so that it can be opened and changed:

| | Greeting over the photos | Three photos and a picture of lettering | Wreaths and a decorated ball |
|---|---|---|---|
| File | `greeting_on_photos_bambu_A1.3mf` | `photos_and_greeting_panel_bambu_A1.3mf` | `wreaths_and_decorated_ball_bambu_A1.3mf` |
| On the pictures | 3 pictures: a line of the greeting below each photo, fairy lights above | 4 pictures: a fir garland below each photo; the fourth is lettering with holly, snowflakes and a garland | 3 pictures: a round wreath, a wreath along the edges, a candy-cane border |
| On the ball | candy-cane stripes between the pictures, a big star underneath | nothing | stars between the pictures, beads round the top, snowflakes round the bottom, a big snowflake underneath |
| Print time (0.4 mm nozzle) | 5 h 40 min | 5 h 25 min | 6 h 29 min |
| Filament | 46 g | 47 g | 47 g |
| Supports, "floating regions" warning | none | none | none |
| Where the layers start | all within 0.8 mm of the seam's centre line | 2 loops of 1,022 at a ridge of a garland, the rest on the line | 2 loops of 1,154 at a ridge, the rest on the line |
| Layers with relief that are a closed ring of two loops | 511 of 511 | 511 of 511 | 577 of 577 |

- One of the automated tests slices an ornament with every kind of decoration at once - lettering
  both high and low, garlands, both wreaths and the candy-cane border on the pictures; stars
  between them, a zigzag round the top, snowflakes round the bottom and a snowflake underneath -
  for both nozzles (6 h 04 min with the 0.4 mm nozzle, 18 h 50 min with the 0.2 mm one). No
  supports, no warning, a closed ring on every layer that carries relief and every wall loop
  started at the back with either.
- Four more decorated balls were sliced while the decorations were being drawn - candy-cane
  stripes and fir branches between the pictures, fir garlands, zigzags and beads round the ball,
  both shapes underneath, relief on Both sides, the 0.2 mm nozzle: the same result, with at most
  6 loops in 2,300 starting at a ridge instead of at the back (see "The seam").
- 150 more, decorated at random over every combination of size, number of pictures, picture
  height, gap and relief side, were built and checked without slicing: every one a closed solid
  with no point that starts in mid-air, and the checklist agreed with the model about what had
  found room in all of them.
- An independent review of the new decorations found ten faults, none of which stopped a print:
  lettering squeezed when a wreath shared its picture with a garland, bands that left a hole
  where they bent sharply (the corners of a zigzag, the point where two swags meet), borders
  too big for narrow pictures, a gap at the back that shrank towards the bottom of the ball,
  and what happens with the relief Inside. All are fixed, and each has a test.
- The plain demo model is unchanged from version 1.4, triangle for triangle: lettering and
  decorations cost nothing where there are none.
- Decorations cost time, because every edge makes the printer slow down and turn: the three
  photos alone slice to 5 h 04 min; with the greeting, the fairy lights, candy-cane stripes
  between them and a star underneath, to 5 h 40 min.

`python tools/gcode_report.py name_bambu_A1.3mf` prints this breakdown for any sliced file.

## Lighting it

Any LED light works: the white inside of the ball evens the light out, so even a single LED at
the cap lights the whole picture. Keep LEDs away from the pictures themselves - one resting
against the wall shows as a bright spot.

Battery lights trade brightness against running time, and only bigger cells buy more of both. A
tree light inside the ball is the yardstick: a few lumens. To keep that up for a season of thirty
six-hour evenings takes roughly the energy of three AA cells; a pair of CR2032 coin cells holds a
tenth of it. No small light gets round that - the ones that run for weeks on one cell do it by
being dim.

| Light | How it goes in | Brightness | One set of cells lasts |
|-------|----------------|------------|------------------------|
| **Tree light** (the tree's own mini LED string) | Push one bulb through the cap opening (preset "Mini string light", 10 mm). | the yardstick | As long as the tree is on - no batteries. |
| **Micro-LED string on a 3 x AA box with a timer** (20 LEDs on 2 m of thin wire; 6 hours on, 18 off) | Through any opening (preset "Fairy lights", 6 mm). Fold the lit end back and forth into a bundle a few centimetres long, twist it so that it holds, and lower it in until it hangs in the middle of the ball. The box (about 5 x 7 x 2 cm, 110 g with cells) stays on a branch. One string can light two or three ornaments that hang close together: a loop of four or five LEDs in each. | brightest battery choice | Sellers give about 48 hours of continuous light, 120 "with breaks": one to three weeks of evenings. Versions with a remote dimmer (mostly longer strings, 50 LEDs and up) last longer turned down. |
| **Coin-cell fairy lights** (2 x CR2032) | As above; the small pack tucks behind the ornament. | bright at first | About a day of light, three or four evenings on a timer. |
| **Puck light** - a "submersible LED tea light", 30 mm, 2 x CR2032, 13 g | Preset "Puck light": it drops into a seat in the top of the cap, lit side down (see below). | bright at first | A few bright hours, then a long dim tail (sellers claim anything from 12 to 150 hours). |
| **LED balloon light** - an 11 mm "bullet" with 3 x LR41 cells, 4 g | Preset "LED balloon light" (11.6 mm). It pushes into the opening until its silicone ring rests on the cap. | bright at first | Two or three hours, then it fades. |
| **LED tea light** - 38 mm; one CR2032, or a CR2450 in the kind with a remote and a timer | Preset "LED tea light": flame down in a seat in the top of the cap. | dim - a fraction of a tree light | 100 hours on a CR2032, a month of evenings on a CR2450 with timer (sellers' figures). |
| **Clip-on LED tree candle** - a 15 mm stick with one AAA cell, remote, dimmer and timer | Custom size, about 16 mm: flame first through the opening; half of the stick stays outside. | dim - a fraction of a tree light | 120 to 150 hours in steady light (three makers' manuals). |

For light that lasts, use the tree's own string where there is one, and the 3 x AA string with a
timer where there is not. Rechargeable AA cells take the sting out of changing them (the light is
a little dimmer on their lower voltage). The dim lights are for a dark room: by a rough
estimate (none of these lights was tried here) the glow they give a lithophane is no brighter
than the tree's own lights reflected off its white surface, so the picture would wash out.

### A light that sits on the cap

**Seat for a light** on the Cap page (the "Puck light" and "LED tea light" presets switch it on)
cuts a round pocket into the top of the cap. The light drops in, lit side down, and shines into
the ball through the light opening, which is kept at least 3 mm narrower than the pocket so that
the light has a ledge to rest on.

- The ledge is a 45 degree cone, not a flat step. Printed cap-down it is the ceiling of the
  pocket, and a cone prints without support; it also centres lights that are a little smaller
  than the pocket.
- The 30 mm puck is not quite a standard: makers give 28.5 to 30.0 mm across and 19 to 26 mm
  tall, and people who measured theirs found 29.0 to 29.6 mm. The preset's 30.6 mm pocket takes
  the largest, 10 mm deep; the 24.5 mm opening holds the smallest. Measure yours and set **Seat
  diameter** to its widest part plus about 0.6 mm if it differs.
- The cap grows to make room: 35.6 mm wide and 13 mm tall for the puck on the 80 mm ball. A tea
  light (39 mm pocket) needs a wider cap than that ball's 30 degree shoulder leaves room for, so
  choosing it raises the shoulder angle to 36 degrees and says so.
- Hang the ornament by a ribbon through **both** loops, so that the ribbon passes over the light
  and keeps it in. A light at the top of a sphere lights the inside nearly evenly: the parts
  of the wall it is nearest to see it at the flattest slant.

## Why it prints without supports

A plain hollow sphere is awkward to print: it touches the bed at a point and its top closes over
empty air. This design removes both problems while still looking like a classic bauble.

- **Printed cap-down.** The cap gives a wide flat footprint (26 mm plus the hanger loops) and the
  loops lie flat on the bed.
- **Shoulder.** Between the cap and the sphere there is a short cone, tangent to the sphere, so the
  outside is never shallower than the shoulder angle (30 degrees by default). At 0.12 mm layers that
  is a sideways step of 0.21 mm per layer - half an extrusion width.
- **Closed end.** The outside is a true sphere. Inside, instead of a dome that gets flatter and
  flatter (which prints over thin air and tends to leave droops or pin-holes), the wall continues as
  a 30 degree cone and finishes with a small flat ceiling that the printer bridges.
- **The two faces of the wall share each edge.** With the default **Outside** setting the picture
  is on the outside. Where the outside faces down and an edge is too sudden for it, the outside
  ramps at 30 degrees from flat - no flatter than the shoulder, 0.21 mm sideways per 0.12 mm
  layer - and the inside, which faces up there, makes up the difference (see "Which side
  carries the picture"). No hidden downward-facing part of the picture is closer to flat than
  20 degrees (0.33 mm per layer), no tip hangs below the wall it belongs to, and the wall keeps
  the thickness the picture asks for.
- **Print-safe edges.** That sharing is what the Print-safe edges switch on the Light & dark page
  does with Outside. Only within about 20 degrees of the middle, where both faces are near their
  limit, is an extreme edge spread over up to half a millimetre (in the demo no thickness is off
  by more than 0.1 mm). With Inside or Both there is one
  relief surface, so there the edge itself is eased. Switched off, the app warns.
- **No shelf beside the pictures** (Inside only). Where a picture edge meets the frame on the
  side that prints as an overhang, the frame ramps over a few millimetres instead of stepping.
  Lit, it shows as a soft fade at that edge of the picture. With Outside the frame stays plain:
  the two faces share that step like any other.
- **Never too thin.** Measured square to its surface, the wall is nowhere thinner than nine
  tenths of the thinnest wall (0.72 mm by default) - wide enough for two lines of the 0.4 mm
  nozzle side by side in every layer (see "Which side carries the picture").
- **Lettering and decorations are picture like any other.** They go through the same sharing of
  edges between the two faces, so nothing of them starts in mid-air either; they are drawn bold,
  and the checklist says when lettering is too small for the nozzle or too fine for where it sits.
- **Seat for a light.** The ledge under a light that sits in the cap is a 45 degree cone, and
  the pocket above it stands on the bed as a rim at least 1.2 mm wide.
- **Checked twice.** On every export the app looks for points that would start in mid-air, and
  Bambu Studio slices the model in the background and reports anything it objects to. The app
  then reads the sliced file: where each layer starts, and whether each layer of the pictures is
  a closed ring of two wall loops.
- **Seam.** A plain divider between two pictures faces the back of the printer, and the project
  marks a line down its middle where every layer starts (see "The seam").
- **Pictures stay clear** of the shoulder and the closed end, so every part of a photo sits on a
  well-supported, steep part of the wall.

## Files

```
main.py                      start the app; `--export out.stl [project]` builds without the window
Lithophane Sphere Studio.bat double-click launcher
lithosphere/
  params.py                  every setting, its limits, and the derived dimensions
  imaging.py                 photo loading, tone, projection onto the sphere
  overlay.py                 lettering and decorations - on the pictures and on the plain
                             parts of the ball - drawn true to the sphere
  fonts.py                   the lettering fonts this computer has
  geometry.py                the watertight mesh
  render.py                  the "glowing" colour model
  checks.py                  printability checklist and mesh verification
  export.py                  STL / 3MF writers, slicer settings, print-settings sheet
  slicer.py                  Bambu Studio hand-off (open, slice, make the project)
  bambu_cli.py               driver for the Bambu Studio command line
  ui/                        the Qt window (ui/tips.py holds every tooltip)
samples/                     four festive sample pictures (made by tools/make_samples.py)
tools/gcode_report.py        what a sliced file will actually print, layer by layer
tests/                       `python -m pytest tests -q`  (set LSPHERE_UI_TEST=1 to include the
                             window tests, LSPHERE_SLICE_TEST=1 for a real Bambu Studio slice)
output/                      the demo ornament, ready to print, and three ornaments made from
                             the photos in my_pics/ (open their .lsphere files to change them)
```
