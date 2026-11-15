"""What every control says when the mouse rests on it.

All the tooltip texts of the window live here, so that they read as one voice
and can be checked as a whole.  Each entry is a title followed by one or more
short paragraphs: what the control does first, then how to choose.  rich()
turns an entry into the small formatted box Qt shows.
"""
from __future__ import annotations

from html import escape

from . import style

WIDTH_PX = 400              # tooltips wrap at this width

TIPS: dict[str, tuple[str, ...]] = {
    # ------------------------------------------------------------------ tabs
    "tab.pictures": (
        "Step 1 - Pictures",
        "Choose how many pictures go round the ornament and which photo each one shows, then "
        "crop and tone each photo.",
        "The steps run from left to right, but you can go back and forth at any time: the "
        "preview on the right follows every change."),
    "tab.decorate": (
        "Step 2 - Decorate",
        "Add a greeting and Christmas decorations to each picture: lettering over a photo, "
        "garlands, wreaths, fairy lights, holly, snowflakes.",
        "A picture can also be lettering only, with no photo at all.",
        "At the bottom of the page are decorations for the rest of the ball: between the "
        "pictures, round the top and the bottom, and underneath."),
    "tab.sphere": (
        "Step 3 - Sphere",
        "The size of the ball, how much of it the pictures cover, the gap between them, and how "
        "a flat photo is laid onto the round surface."),
    "tab.tones": (
        "Step 4 - Light & dark",
        "How thick the wall is where a picture is bright and where it is dark. That sets the "
        "contrast of the lit ornament, and also its strength and weight."),
    "tab.cap": (
        "Step 5 - Cap",
        "The cap on top of the ornament: the opening or seat for the light, the size of the "
        "cap, and the loops the ornament hangs by."),
    "tab.print": (
        "Step 6 - Print",
        "Choose the nozzle, read the checklist - it says in plain words whether the ornament "
        "will print well and look right - and see the Bambu Studio settings.",
        "When the checklist is green, press Export for printing."),

    # -------------------------------------------------------------- pictures
    "count.1": (
        "One picture",
        "A single picture wraps almost all the way round the ball. Good for a panorama, a "
        "pattern or one long line of lettering; a portrait would be stretched round the sides."),
    "count.2": (
        "Two pictures",
        "One picture on the front and one on the back, each nearly half the ball wide. The "
        "outer parts of each curve away from you, so keep faces near the middle."),
    "count.3": (
        "Three pictures",
        "Three pictures, each a third of the way round. From any side you see one whole "
        "picture, almost square. The usual choice for an 80 mm ornament."),
    "count.4": (
        "Four pictures",
        "Four narrower, upright pictures. Portrait photos suit them; a wide photo loses its "
        "sides unless you zoom out. A good way to have three photos and one picture of "
        "lettering only."),
    "card": (
        "Picture {n}",
        "Click to select this picture. The sliders below then adjust it, the Decorate page "
        "decorates it, and the preview turns to face it.",
        "You can drop a photo here from Explorer. Dropping several at once fills this picture "
        "and the ones after it."),
    "card.choose": (
        "Choose photo",
        "Pick the photo for this picture. JPG, PNG, WEBP, TIFF, BMP and GIF all work.",
        "The ornament shows a picture as light and dark only, so colour is turned into shades "
        "of grey: bright parts of the photo become thin wall and glow, dark parts become thick "
        "wall and stay dark. Photos with clear faces and good contrast work best."),
    "card.remove": (
        "Remove this photo",
        "Takes the photo out of this picture. Its lettering and decorations stay - with those "
        "alone the picture becomes a plain panel that carries just the lettering."),
    "zoom": (
        "Zoom",
        "How large the photo is in its picture. At 1.00 the photo just fills the picture.",
        "Above 1 it is enlarged and more is cut off; use that to close in on faces. Below 1 "
        "the photo shrinks and leaves a plain border, which keeps all of a wide photo and "
        "makes room for lettering above or below it.",
        "The photo settings are greyed out while the selected picture has no photo."),
    "shift_x": (
        "Move left / right",
        "Slides the photo sideways behind its picture, to choose which part is cut off.",
        "It can only move as far as there is spare photo. If the photo exactly fits in this "
        "direction nothing happens - zoom in first."),
    "shift_y": (
        "Move down / up",
        "Slides the photo down or up behind its picture, to choose which part is cut off.",
        "It can only move as far as there is spare photo. If the photo exactly fits in this "
        "direction nothing happens - zoom in first."),
    "rotate_left": (
        "Rotate left",
        "Turns the photo a quarter turn anticlockwise. For photos that arrive lying on their "
        "side."),
    "rotate_right": (
        "Rotate right",
        "Turns the photo a quarter turn clockwise. For photos that arrive lying on their side."),
    "mirror": (
        "Mirror",
        "Flips the photo left to right, as in a mirror. Any writing inside the photo then reads "
        "backwards; lettering you add on the Decorate page is not affected."),
    "brightness": (
        "Brightness",
        "Lightens or darkens the whole photo. Lighter means thinner walls and more glow.",
        "Small steps go a long way. If only the middle greys look too dark, use Mid-tone "
        "brightness on the Light & dark page instead - it leaves black and white alone."),
    "contrast": (
        "Contrast",
        "Above 1 pushes light and dark further apart: more punch, but the very brightest and "
        "darkest parts lose their detail. Below 1 flattens the photo."),
    "auto_levels": (
        "Auto levels",
        "Stretches the photo's tones so that its darkest part becomes the thickest wall and its "
        "brightest part the thinnest. A hazy or dull photo then uses the full range the "
        "ornament can show.",
        "Leave it on unless you want a picture that is pale or dark on purpose."),
    "reset_slot": (
        "Reset this picture's adjustments",
        "Puts zoom, position, rotation, mirror, brightness, contrast and Auto levels of the "
        "selected picture back to where they started.",
        "The photo itself, its lettering and its decorations stay."),

    # -------------------------------------------------------------- decorate
    "deco.pick": (
        "Picture {n}",
        "Lettering and decorations belong to one picture. Choose here which one you are "
        "working on - the preview turns to face it.",
        "It is the same selection as clicking a picture on the Pictures page."),
    "text": (
        "Lettering",
        "Type what the picture should say - for example Happy Holidays. Press Enter to start a "
        "new line; up to six lines.",
        "The whole block is fitted into the picture, so the longest line decides how large "
        "everything can be. A long greeting reads best broken into three or four short lines.",
        "With no photo chosen, the picture becomes a plain panel that carries only this "
        "lettering.",
        "The settings under the box are greyed out until there is some lettering to set."),
    "font": (
        "Font",
        "The typeface of the lettering. The fonts at the top of the list are sturdy, festive "
        "ones that print well; below the line are all the other fonts on this computer. A font "
        "you install in Windows appears here the next time the app starts.",
        "A printed letter is only as good as its thinnest stroke. Bold, rounded fonts stay "
        "readable when small; fine script fonts need to be large, or thickened with Thicken "
        "letters."),
    "tone.light": (
        "Glowing letters",
        "The letters are the thinnest wall, so they light up. A dark edge round them keeps them "
        "readable on any background. The usual choice."),
    "tone.dark": (
        "Dark letters",
        "The letters are the thickest wall and stay dark, with a light edge round them. Good on "
        "a very bright photo, or on a Bright panel."),
    "text_size": (
        "Size",
        "How large the lettering is. 100 % is the largest at which the whole block still fits "
        "inside the picture.",
        "The line under the text box says how tall the letters come out in millimetres and how "
        "wide their strokes are. If one long line makes everything small, break it in two."),
    "text_x": (
        "Move left / right",
        "Slides the lettering sideways in whatever room it has - none, when the longest line "
        "already fills the width of the picture."),
    "text_y": (
        "Move down / up",
        "Slides the lettering towards the bottom or the top of the picture.",
        "Lettering is sharpest below the middle. The upper half of every picture is printed "
        "with its outside facing down, and there the printer cannot build crisp edges: fine "
        "letters placed high up come out soft. The Glowing view shows how soft to expect, and "
        "the checklist warns when it matters.",
        "Towards the top and the bottom the picture is also narrower, so a wide block shrinks "
        "to stay inside it."),
    "text_outline": (
        "Edge round the letters",
        "Width of the contrasting edge drawn round every letter: dark round glowing letters, "
        "light round dark ones. It keeps the lettering readable against the photo behind it.",
        "0 leaves it out, which looks cleanest on a plain panel."),
    "text_bold": (
        "Thicken letters",
        "Makes every stroke this much wider on each side. It turns a fine or script font into "
        "one sturdy enough to print.",
        "The checklist on the Print page warns when the strokes are too thin for the nozzle."),
    "deco_top": (
        "Along the top",
        "A decoration hung along the top edge of this picture.",
        "The top of a picture is printed with its outside facing down, where the printer cannot "
        "build crisp edges, so shapes up there always come out softer than in the lower half. "
        "These decorations are drawn bold for that reason; the Glowing view shows the result."),
    "deco_bottom": (
        "Along the bottom",
        "A decoration along the bottom edge of this picture. The lower part of a picture is "
        "where everything prints sharpest."),
    "deco_frame": (
        "All the way round",
        "A border round the whole picture.",
        "A border runs right along the edge of the picture: what it covers is hidden, and no "
        "sliver of the photo is left outside it.",
        "The wreath comes in two shapes. The round one turns the picture into a round window "
        "and leaves the corners plain; the other follows the edges of the picture, so only the "
        "border of the photo is covered.",
        "The top of any border comes out softer than its bottom, like everything in the upper "
        "half of a picture."),
    "deco_scatter": (
        "Scattered",
        "Small shapes sprinkled over the free parts of the picture.",
        "Over a photo they keep clear of the middle, where the faces usually are. They also "
        "keep off the lettering and the other decorations. Shapes in the upper half come out "
        "softer than those lower down."),
    "deco.none": ("Nothing", "No decoration here."),
    "deco.garland": (
        "Fir garland with baubles and bows",
        "A garland of fir hung in swags: a dark bough thick with glowing needles, round "
        "baubles along it and a bow where each swag is tied up.",
        "Along the top the needles are drawn bolder, because fine ones would not show there."),
    "deco.lights": (
        "String of fairy lights",
        "A wire draped across the top with a row of glowing bulbs hanging from it."),
    "deco.icicles": (
        "Snow and icicles",
        "A glowing rim of snow along the top edge with icicles of different lengths hanging "
        "from it."),
    "deco.holly": (
        "Holly in the corners",
        "A sprig of three holly leaves with berries in each corner of this edge. Leaves the "
        "middle of the edge free."),
    "deco.snow": (
        "Bank of snow",
        "A glowing bank of snow along the bottom edge, like a snowy foreground."),
    "deco.trees": (
        "Snow with fir trees",
        "A bank of snow with dark fir trees standing in it, taller towards the sides so that "
        "they frame the picture."),
    "deco.line": (
        "Line frame",
        "A thin glowing line with rounded corners, right at the edge of the picture."),
    "deco.candy": (
        "Candy-cane border",
        "A broad glowing band right round the edge of the picture, with bold dark stripes on "
        "the slant - one wide, one narrow, turn and turn about - like a stick of candy."),
    "deco.wreath": (
        "Round wreath",
        "A Christmas wreath: a ring of fir thick with glowing needles, baubles all round it "
        "and a bow at the foot.",
        "The photo shows inside the ring and everything outside it becomes plain, so keep the "
        "faces near the middle of the picture. Lettering is fitted inside the ring, and "
        "scattered shapes go into the corners round it."),
    "deco.wreath_square": (
        "Wreath along the edges",
        "The same fir wreath with baubles and a bow, but laid along the edges of the picture "
        "instead of in a ring.",
        "All of the photo shows except the border the wreath covers - the choice for a photo "
        "that fills its picture."),
    "deco.snowflakes": ("Snowflakes", "Six-armed snowflakes of different sizes."),
    "deco.snowdots": ("Falling snow", "Round flakes of falling snow, many and small."),
    "deco.stars": ("Stars", "Five-pointed stars of different sizes."),
    "deco.hearts": ("Hearts", "Hearts of different sizes."),
    "deco_amount": (
        "How many",
        "How many shapes are scattered over the picture, from a few to a flurry.",
        "Greyed out until a shape is chosen under Scattered."),
    "deco_size": (
        "Size of the decorations",
        "Makes every decoration of this picture larger or smaller. At 1.00 they are in "
        "proportion to the picture.",
        "Smaller looks daintier but prints less clearly: shapes under about a millimetre across "
        "are lost, most of all near the top of the picture.",
        "A wreath or a candy-cane border stops growing where it would swallow the picture: on "
        "the narrow pictures of a ball with four, it is drawn smaller than this asks.",
        "Greyed out until this picture has a decoration."),
    "shuffle": (
        "Shuffle",
        "Deals the scattered shapes afresh. Click until you like how they fall.",
        "Greyed out until a shape is chosen under Scattered."),
    "backdrop.plain": (
        "Plain panel",
        "For a picture without a photo. The panel is as thick as the frame, so it blends into "
        "the ornament and the lettering looks cut into the ball itself."),
    "backdrop.dark": (
        "Dark panel",
        "For a picture without a photo. The panel is the thickest wall: glowing letters and "
        "decorations stand out most strongly against it."),
    "backdrop.light": (
        "Bright panel",
        "For a picture without a photo. The panel is the thinnest wall and glows all over. "
        "Decorations are then drawn dark; choose Dark letters to go with it."),
    "copy_deco": (
        "Use these decorations on every picture",
        "Copies this picture's decorations - top, bottom, border, scattered shapes and their "
        "size - to all the other pictures, so that they match.",
        "Lettering is not copied."),
    "clear_deco": (
        "Remove lettering and decorations",
        "Takes the lettering and every decoration off this picture. The photo and its "
        "adjustments stay."),
    "ball_dividers": (
        "Between the pictures",
        "A decoration on the plain dividers between the pictures, running from top to bottom.",
        "It belongs to the ball itself, not to any one picture. It is made the way the "
        "pictures are: with them on the outside of the wall (the usual setting), it is cut "
        "into the outside below the middle of the ball and prints crisp there, and comes out "
        "softer higher up.",
        "The divider at the back stays plain. The printer starts every layer there, and that "
        "seam hides best in plain, thick plastic. With one picture it is the only divider.",
        "A decoration needs a divider wide enough to hold it. If it does not fit, the checklist "
        "says how wide to make Gap between pictures on the Sphere page."),
    "ball_top": (
        "Round the top",
        "A ring of decoration round the ball, between the pictures and the cap.",
        "Up there the outside of the ball faces down while it prints and cannot carry a "
        "pattern, so this one is formed on the inside of the wall. It shows softly when the "
        "ornament is lit, and not at all when it is not.",
        "The ring leaves a small gap at the back, where the seam runs. If there is too little "
        "plain ball above the pictures, the checklist says how far to lower Picture height on "
        "the Sphere page."),
    "ball_bottom": (
        "Round the bottom",
        "A ring of decoration round the ball below the pictures.",
        "This part prints crisp: the pattern is cut into the outside of the ball and shows "
        "both lit and unlit.",
        "The ring leaves a small gap at the back, where the seam runs. If there is too little "
        "plain ball below the pictures, the checklist says how far to lower Picture height on "
        "the Sphere page.",
        "With Inside chosen on the Sphere page there can be no ring down here: that strip is "
        "then where the inside of the wall runs in to the closed end."),
    "ball_pole": (
        "Underneath",
        "One large shape on the very bottom of the ball - what you see when the ornament hangs "
        "above you.",
        "It stands a little proud of the ball instead of being cut into it: the wall down "
        "there closes the ball and is not made thinner. Lit, it shows as a darker shape on the "
        "glowing bottom."),
    "ball.none": ("Nothing", "This part of the ball stays plain."),
    "ball.stars": (
        "Stars",
        "Glowing five-pointed stars: a column of them up every divider, or a row of them "
        "round the ball."),
    "ball.snowflakes": (
        "Snowflakes",
        "Glowing six-armed snowflakes: a column of them up every divider, or a row of them "
        "round the ball."),
    "ball.beads": (
        "String of beads",
        "Round glowing beads in a line, like a string of pearls on a tree."),
    "ball.candy": (
        "Candy-cane stripes",
        "Turns each divider into a stick of candy: a glowing band with bold dark stripes on "
        "the slant."),
    "ball.fir": (
        "Fir",
        "Fir thick with glowing needles, with a few baubles: a branch up every divider, or a "
        "garland round the ball.",
        "It needs more room than the others: a gap of about 12 degrees between the pictures on "
        "an 80 mm ball, or about 8 mm of plain ball above or below them."),
    "ball.zigzag": (
        "Zigzag",
        "A glowing zigzag line round the ball, as on a knitted Christmas jumper."),
    "ball.snowflake": (
        "Big snowflake",
        "One large six-armed snowflake centred on the bottom of the ball."),
    "ball.star": (
        "Big star",
        "One large five-pointed star centred on the bottom of the ball."),

    # ---------------------------------------------------------------- sphere
    "diameter": (
        "Sphere diameter",
        "The outside diameter of the ball. 80 mm is a classic large bauble.",
        "A bigger ball shows more detail of the photos and takes larger lettering, but it takes "
        "longer to print and weighs more: the checklist says when it gets too heavy for a "
        "branch. The cap grows and shrinks with the ball."),
    "band_height_deg": (
        "Picture height",
        "How far the pictures reach towards the top and the bottom of the ball, as an angle. "
        "The areas above and below them stay plain - or carry a ring of decoration, chosen at "
        "the bottom of the Decorate page; lower pictures leave more room for one.",
        "Taller pictures show more of a photo, but their top and bottom edges curve away from "
        "the viewer. The slider stops where the pictures would run into the cap's shoulder or "
        "the closed end."),
    "gap_deg": (
        "Gap between pictures",
        "The width of the plain divider between neighbouring pictures.",
        "The printer starts every layer at the same place, which leaves a faint line - the "
        "seam. The exported Bambu Studio project puts it down the middle of the divider at the "
        "back, so the gap has to be wide enough to hold it: about 3 degrees on an 80 mm ball. "
        "The checklist says if it is too narrow. Beyond that, the gap is simply how much frame "
        "you like between the pictures."),
    "projection.natural": (
        "Natural",
        "Lays the photo on the ball so that shapes and faces keep their proportions "
        "everywhere. The picture's corners are trimmed a little, because a flat photo cannot "
        "cover a round surface exactly. Recommended."),
    "projection.wrap": (
        "Wrap",
        "Keeps the whole photo, edge to edge, by squeezing it narrower towards the top and the "
        "bottom of the ball. Use it when nothing may be cut off. A single picture that goes all "
        "the way round always uses Wrap."),
    "relief.outside": (
        "Picture on the outside",
        "The thick and thin of the picture is shaped on the outside of the wall. Lit from "
        "within, that looks sharpest, because the light passes the picture last. Recommended.",
        "The ornament is printed upside-down, so in the half nearest the cap the outside faces "
        "down. Where an edge there is too sudden to print, the inside of the wall carries part "
        "of it - nothing hangs over thin air, but such edges look a little softer."),
    "relief.split": (
        "Picture on both sides",
        "The upper half of each picture is shaped on the inside of the wall, the lower half on "
        "the outside. The ornament feels smooth in its upper half.",
        "Lit, the half shaped on the inside looks soft: the light spreads sideways in the "
        "plastic after passing it, and small details wash out."),
    "relief.inside": (
        "Picture on the inside",
        "The whole picture is shaped on the inside of the wall. The outside of the ornament is "
        "a smooth ball.",
        "Lit, the pictures look soft all over: the light spreads sideways in the plastic after "
        "passing them. The Glowing view shows the difference."),
    "reset_page": (
        "Reset this page",
        "Puts every setting on this page back to its recommended value. Other pages are not "
        "touched."),

    # ----------------------------------------------------------------- tones
    "t_min": (
        "Thinnest wall (brightest)",
        "The wall thickness where a picture is white. Thinner glows brighter but is more "
        "fragile.",
        "0.8 mm is a good choice. Below about 0.75 mm the 0.4 mm nozzle can no longer lay two "
        "lines side by side, so the brightest areas print as a single weak line - the "
        "checklist says so. The 0.2 mm nozzle can go thinner."),
    "t_max": (
        "Thickest wall (darkest)",
        "The wall thickness where a picture is black. Thicker gives deeper shadows and more "
        "contrast, but with a weak light the whole picture may look dim.",
        "It stays at least 0.5 mm above the thinnest wall. Around 2.8 mm suits a tree light."),
    "t_frame": (
        "Frame thickness",
        "The thickness of everything that is not picture: the dividers, the plain top and "
        "bottom, and a Plain panel of lettering.",
        "Thicker makes a darker frame and a stronger, heavier ornament. Between the thinnest "
        "and the thickest wall the frame glows softly; make it as thick as the darkest parts if "
        "the pictures should stand out more."),
    "gamma": (
        "Mid-tone brightness",
        "Above 1 lifts the middle greys without changing pure black or white - good for photos "
        "that look heavy when lit. Below 1 deepens them."),
    "smoothing": (
        "Smoothing",
        "Softens grain and fine noise in the photos, which would otherwise print as speckle. "
        "0 keeps every detail.",
        "Lettering and decorations are not smoothed: they stay crisp whatever this is set to."),
    "safe_edges": (
        "Print-safe edges",
        "Makes sure no part of a picture starts in mid-air. Where the relief hangs from a "
        "leaning wall, a sudden edge would print as a blob - Bambu Studio calls those floating "
        "regions. With this on, such an edge gets the steepest slope that still prints, and "
        "everything else stays exactly as it is.",
        "Leave it on. Switched off, the checklist warns and Bambu Studio will too."),
    "invert": (
        "Negative",
        "Swaps bright and dark in the photos, so the lit picture becomes a negative. Only for "
        "special effects. Lettering and decorations are not swapped."),

    # ------------------------------------------------------------------- cap
    "light_preset": (
        "Light",
        "Choose the light that goes into the ornament, and the cap is made to fit it: an "
        "opening the bulb passes through, or a seat for a light that sits on top.",
        "Measure your own light before printing - makes differ by a millimetre or so - and "
        "correct the sizes below if needed. Use LED lights only: other bulbs get hot enough to "
        "soften the plastic."),
    "light.mini": (
        "Mini string light",
        "A bulb of an ordinary LED tree-light string, pushed through a 10 mm opening. Needs no "
        "batteries, and it is as bright as the rest of the tree."),
    "light.fairy": (
        "Fairy lights",
        "Micro LEDs on thin wire, fed through a 6 mm opening. Fold the lit end into a small "
        "bundle that hangs in the middle of the ball - an LED resting against a picture shows "
        "as a bright spot."),
    "light.balloon": (
        "LED balloon light",
        "A small battery light shaped like a bullet. It is pushed into an 11.6 mm opening "
        "until its ring rests on the cap. Bright, but runs only a few hours."),
    "light.puck": (
        "Puck light, 30 mm",
        "A round battery light that drops into a seat on top of the cap, lit side down. Bright "
        "for some hours on its coin cells, then dim."),
    "light.tealight": (
        "LED tea light, 38 mm",
        "An LED tea light that sits in a seat on top of the cap, flame down. It runs for a "
        "hundred hours and more, because it is dim. On an 80 mm ball the cap's shoulder is made "
        "a little steeper to make room."),
    "light.c6": ("C6 LED bulb", "A 19 mm bulb through a 20.5 mm opening. LED versions only."),
    "light.c7": ("C7 LED bulb", "A 22 mm bulb through a 24 mm opening. LED versions only."),
    "light.c9": ("C9 LED bulb", "A 29 mm bulb through a 30.5 mm opening. LED versions only."),
    "light.custom": (
        "Custom size",
        "Any other light: set the opening - and the seat, if the light sits on top - yourself "
        "with the sliders below."),
    "hole_diameter": (
        "Light opening",
        "The diameter of the hole in the top of the cap that the bulb goes through. Measure "
        "your bulb and add about 1 mm.",
        "With a seat, this is the window the light on top shines through. It stays at least "
        "3 mm narrower than the seat, so that the light has a ledge to rest on."),
    "seat": (
        "Seat for a light that sits on top",
        "Cuts a round pocket into the top of the cap. A battery puck or LED tea light drops "
        "into it, lit side down, and shines into the ball through the light opening.",
        "Hang the ornament by a ribbon through both loops, so that the ribbon passes over the "
        "light and keeps it in."),
    "seat_diameter": (
        "Seat diameter",
        "The diameter of the pocket the light drops into. Measure your light across its widest "
        "part and add about 0.6 mm.",
        "Only used when Seat for a light is ticked. The light opening and the cap follow it."),
    "seat_depth": (
        "Seat depth",
        "How deep the light sits in the cap. A third of the light's height is enough to hold "
        "it; deeper hides more of it. The cap is made taller if it has to be.",
        "Only used when Seat for a light is ticked."),
    "cap_diameter": (
        "Cap diameter",
        "The width of the round cap on top, like the metal cap of a glass bauble.",
        "It is also the foot the ornament stands on while it prints, so wider is steadier. The "
        "slider stops at the narrowest cap that still surrounds the opening and the widest "
        "that still fits under the ball's shoulder."),
    "cap_height": (
        "Cap height",
        "How tall the cap is. It is made taller on its own when a deep seat needs the room."),
    "cap_wall": (
        "Cap wall",
        "The thickness of the cap's side wall and of the plate around the light opening. 2 mm "
        "is sturdy; there is rarely a reason to change it."),
    "flare_angle": (
        "Shoulder angle",
        "The ball runs into the cap through a short cone - the shoulder - so that the ornament "
        "prints without supports. This is the cone's angle from flat.",
        "Lower looks rounder; higher prints more easily and leaves room for a wider cap. "
        "30 to 35 degrees is a good range."),
    "hanger": (
        "Hanger",
        "The loops on the cap that the ornament hangs by. They lie flat on the printer's bed "
        "while it prints, so they cost nothing and help it stick."),
    "hanger.two": (
        "Two loops",
        "One loop on either side of the cap. A ribbon or hook through both hangs the ornament "
        "straight, and passes over a light sitting in a seat."),
    "hanger.one": ("One loop", "A single loop. The ornament hangs slightly tilted from it."),
    "hanger.none": (
        "No loops",
        "No hanger at all: the ornament sits on its light or between branches."),
    "hanger_hole": (
        "Hanger hole",
        "The diameter of the hole in each loop, for a ribbon, thread or ornament hook. Holes "
        "under 2.5 mm tend to close up when printed."),

    # ----------------------------------------------------------------- print
    "nozzle.0.4": (
        "0.4 mm nozzle",
        "The nozzle the A1 comes with. An 80 mm ornament prints in about five hours."),
    "nozzle.0.2": (
        "0.2 mm nozzle",
        "Lines half as wide in layers half as tall: finer detail in the pictures and finer "
        "lettering, for about three times the print time.",
        "Fit the nozzle and change the nozzle size on the printer's screen before printing. "
        "The exported project already selects the matching Bambu Studio presets."),
    "quality": (
        "Model detail",
        "How finely the exported model follows the photos. Standard already holds more detail "
        "than the chosen nozzle can print.",
        "The preview is drawn at its own resolution; this only changes the exported file."),
    "quality.draft": (
        "Draft",
        "A smaller file that exports and slices quickly. For trying things out; fine detail is "
        "coarser."),
    "quality.standard": (
        "Standard",
        "Recommended. More detail than the nozzle can reproduce, in a file of sensible size."),
    "quality.fine": (
        "Fine",
        "The finest model. The file is large and slicing is slower; the print rarely looks "
        "different from Standard."),
    "project_check": (
        "Ready-to-print Bambu Studio project",
        "On export, Bambu Studio slices the ornament in the background. That proves it slices "
        "cleanly, tells you the real print time and filament, and saves a project with the "
        "printer, filament and every setting already chosen - open it, press Slice plate, then "
        "Print plate.",
        "Takes under a minute; a minute or two for the 0.2 mm nozzle, whose model is finer."),
    "check.ok": (
        "Fine",
        "This point has been checked and needs nothing from you."),
    "check.warn": (
        "Worth a look",
        "The ornament will print, but this may not turn out the way you want. The text says "
        "what to change if you care."),
    "check.error": (
        "Must be fixed",
        "The ornament cannot be printed as it is. The text says which setting to change."),
    "settings": (
        "Bambu Studio setting",
        "These are already inside the ready-to-print project. They are listed in case you slice "
        "the plain model yourself."),
    "guide": (
        "Printing guide",
        "Opens the full guide for this ornament: every Bambu Studio setting with its reason, "
        "what to do after printing, and notes on battery lights. The same text is saved next to "
        "the exported model."),
    "export": (
        "Export for printing",
        "Builds the full-detail model and saves it: an STL or 3MF file, a sheet of print "
        "settings and - if ticked on the Print page - a ready-to-print Bambu Studio project.",
        "The model is already turned the right way up for printing, cap on the bed."),

    # --------------------------------------------------- preview, status bar
    "mode.lit": (
        "Glowing",
        "How the ornament looks on the tree with a light inside. Thin walls glow, thick walls "
        "stay dark, and parts that print soft are shown soft - what you see here is close to "
        "the finished print.",
        "Click again to reset the view."),
    "mode.unlit": (
        "Unlit",
        "The plain white print in daylight, when no light shines through: you see the picture "
        "only as relief.",
        "Click again to reset the view."),
    "mode.cutaway": (
        "Cutaway",
        "The ornament sliced in half through the selected picture, to show how thick the wall "
        "is everywhere and how the cap is built.",
        "Click again to reset the view."),
    "mode.bed": (
        "On the printer",
        "The ornament upside-down on the print bed, exactly as it is exported: cap down, "
        "closed end up.",
        "Click again to reset the view."),
    "light_colour": (
        "Colour of the light",
        "The light used in the Glowing view: warm white like a classic tree light, or cool "
        "white. It only changes the preview, not the print."),
    "summary": (
        "Size and weight",
        "Width and height of the printed ornament, loops included, and how much plastic it "
        "takes."),
    "verdict": (
        "Print check",
        "The result of the checklist in a word. Click to open the Print page and read the "
        "details."),
    "mouse": (
        "Moving the preview",
        "Drag to turn the ornament, scroll to zoom, Shift-drag to move it. Home resets the "
        "view."),
    "look": (
        "Look at picture {n}",
        "Turns the ornament to face picture {n} and selects it for the sliders."),
    "spin": (
        "Spin",
        "Slowly turns the ornament so that you see every side. Click again to stop."),

    # ------------------------------------------------------------------ menu
    "menu.new": (
        "New ornament",
        "Starts again with the built-in sample pictures and the recommended settings."),
    "menu.open": (
        "Open project",
        "Opens an ornament saved earlier (a .lsphere file) with all its settings."),
    "menu.save": (
        "Save project",
        "Saves the settings of this ornament so that you can carry on later. The photos "
        "themselves are not copied - keep them where they are."),
    "menu.save_as": ("Save project as", "Saves this ornament under a new name."),
    "menu.exit": (
        "Exit",
        "Closes the app. What you were working on comes back the next time you start it."),
    "menu.reset_view": ("Reset view", "Points the preview back at the selected picture."),
    "menu.about": ("About", "Which version of the app this is."),
    "menu.tips": (
        "Show tips",
        "Shows or hides these explanations when the mouse rests on a control."),
    "box.open": (
        "Open in Bambu Studio",
        "Opens the ready-to-print project in Bambu Studio. There: Slice plate, then Print "
        "plate."),
    "box.folder": ("Show in folder", "Opens the folder that holds the exported files."),
    "guide.copy": ("Copy to clipboard", "Copies the whole guide, to paste it somewhere else."),
}

_used: set[str] = set()


def rich(title: str, *paragraphs: str, foot: str = "") -> str:
    """A tooltip as the rich text Qt shows: title, paragraphs, quiet footer."""
    parts = []
    if title:
        parts.append(f'<p style="margin:0 0 4px 0;"><b><span style="color:{style.ACCENT};">'
                     f'{escape(title, quote=False)}</span></b></p>')
    for text in paragraphs:
        parts.append(f'<p style="margin:0 0 5px 0;">{escape(text, quote=False)}</p>')
    if foot:
        parts.append(f'<p style="margin:0; color:{style.MUTED};">{escape(foot, quote=False)}</p>')
    return (f'<table width="{WIDTH_PX}" cellspacing="0" cellpadding="0"><tr><td>'
            f'{"".join(parts)}</td></tr></table>')


def tip(key: str, foot: str = "", **values) -> str:
    """The tooltip for a control.  values fill the {placeholders} of its text."""
    entry = TIPS[key]
    _used.add(key)
    return rich(*(text.format(**values) if values else text for text in entry), foot=foot)


def body(key: str) -> tuple[str, ...]:
    """The plain paragraphs of an entry, without its title."""
    _used.add(key)
    return TIPS[key][1:]


def unused() -> set[str]:
    """Entries no control has asked for (so far): a spelling slip, or a tip left behind."""
    return set(TIPS) - _used
