# WeftBench

WeftBench is a local fabric-image quality bench. It keeps original photographs, working views, calibration records, drawn observations, retained measurement maps, and portable review evidence in one library.

## Run locally

Run `./start.sh` and open [WeftBench](http://127.0.0.1:19954). The Compose service binds to loopback and keeps its SQLite library and image assets in the named `mill-library` volume. Run `docker compose stop` to stop the service and preserve the library. Set `WEFTBENCH_PORT` when a different local port is needed.

For a Python-only launch, use Python 3.11 and install the versions in `requirements.txt`. Run `python -m weftbench` and open port 8000. Set `WEFTBENCH_DATA` to keep the library in a chosen directory.

## First inspection

1. Select **Load training fabrics** or use **Add fabric** and **Import photo**. The Juniper and Bluestone training collections are synthetic. Their photographs and dimensions illustrate the tools.
2. Open an incoming face photograph at the inspection bench. The coordinate selector switches between the saved physical plane and the original view. Use **Calibrate** to place four clockwise corners on a new source and save a plane.
3. Select **Warp and weft pitch** and run or save the reading. Use **Observe** to draw a polygon or a sewn trace. Use **Sample pixel** for a local image-color and support reading.
4. Choose a transmitted-light photograph for opening candidates or relative light passage. Choose a raking-light photograph for relief and optical abrasion. Select a compatible second view for paired inspection. When the lamp direction changed between the two views, use **Reference-corrected difference** and circle stable areas as illumination references; areas the references cannot vouch for are marked as not comparable.
5. Use **Details** to save picture crops with source location, coverage maps, and native working-pixel downloads. **Print** makes a cropped physical proof or fit-to-page PDF.
6. Open the **Evidence desk** to review prior observations and measurements, approve a color reading, record a visit, assemble a contact sheet, or create and verify an evidence ZIP. File links download the retained bytes.

## Image inputs and limits

- Source intake accepts one PNG, JPEG, TIFF, or WebP photograph per record, up to 24 MiB. Working views apply EXIF orientation, convert available color profiles to sRGB, and use at most 5.2 million pixels or 2700 pixels on either side. The original download keeps its original bytes.
- Per fabric the limits are 120 source captures and 120 retained instrument readings. Detail sheets accept up to six crops. Each crop uses at most two million working pixels, with up to 5.2 million across a sheet.
- Measurements describe visible image evidence under the recorded role and calibration. The pitch spectrum reports optical candidates. It does not classify a weave. Color samples describe image color under recorded illumination. Treat transparency, folds, severe shadows, and clipped highlights as limited evidence.
- Calibrated millimetres describe the chosen plane. Crops printed at a requested scale must fit the selected sheet. Use the PDF viewer’s actual-size print mode for physical checks.
- Intake preview checks exact original bytes and compares recent thumbnails under the selected light role. Visual resemblance is a review cue. Portable evidence verifies the recorded files and their declared hashes on receipt.
