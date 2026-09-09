# Demo — A15 GAN Handwritten Digit Generation

A short script that loads the **already-trained** model and runs the Generator and
Discriminator live. Nothing is trained here, so it finishes in a few seconds.

## How to run

From the project root:

```
venv\Scripts\python.exe demo\demo.py
```

Optional flags:

| Flag | Effect |
|---|---|
| `--n 32` | generate 32 digits instead of 16 |
| `--arch` | also print the full layer-by-layer structure of both networks |
| `--no-open` | don't pop the images open at the end |

It writes three images into `demo/demo_outputs/` and opens them automatically.
It works without internet: MNIST is already in `data/`, and the weights are read
from `Trained model/model networks/`.

## What it shows, in order

1. **Loads the trained model.** Prints the parameter counts (1,935,808 and 659,457),
   which match Table 6 of the report, proving these are the submitted weights.
2. **Generator:** feeds random noise in, gets digits out. Prints the shapes so you
   can literally point at `(16, 100) → (16, 1, 28, 28)`.
3. **Discriminator:** scores 8 real MNIST images and 8 generated ones, prints a
   table, then an average over 256 of each. Saves a picture with each image
   labelled by its own score.
4. **Latent space:** shows the same noise always gives the same digit, then morphs
   one digit into another to show the space is smooth.

## The one-line explanation of each network

- **Generator** takes 100 random numbers and turns them into a 28×28 image. It
  never saw a real MNIST image; it only learned from the Discriminator's feedback.
- **Discriminator** takes an image and outputs one number between 0 and 1: how
  confident it is that the image is real rather than generated.
- They train against each other: G tries to fool D, D tries not to be fooled.

## Likely questions and short answers

**"Why is D(x) only about 0.55 here when your report says 0.87?"**
The report's numbers were logged *during training*, where BatchNorm normalises
using each batch's own statistics. The demo runs in `eval()` mode, the correct
setting for inference, where BatchNorm uses the running averages saved inside the
`.pth` file. Same weights, softer scores. Flipping this same checkpoint back to
`train()` mode gives about 0.81 / 0.12, which matches the report. The script prints
this explanation on screen too.

**"Why isn't it at 0.5, the equilibrium?"**
The Discriminator stayed ahead of the Generator for all 50 epochs. This is common
for a DCGAN on MNIST and doesn't mean generation failed — BCE loss correlates
poorly with image quality (Salimans et al., 2016). Section 6.2 of the report covers it.

**"Is it just memorising the training images?"**
No. The latent walk in step 4 morphs smoothly between two random noise vectors, and
every in-between frame is still a plausible digit. A memorising model would jump
abruptly between stored images.

**"Which digit will it generate?"**
We can't choose. Each noise vector maps to whatever digit the Generator associated
with that region of the space. Choosing the digit requires a Conditional GAN, which
is listed in our future scope.

**"Show me the architecture."**
Run it with `--arch`, or point at Tables 4 and 5 in the report. Generator: a Linear
projection to 256×7×7, then two ConvTranspose2d blocks (7→14→28) with BatchNorm+ReLU,
then Conv2d + Tanh. Discriminator: three strided Conv2d blocks (28→14→7→3) with
LeakyReLU(0.2), then Flatten + Linear + Sigmoid.

**"Why `.detach()` in the training loop?"**
When updating the Discriminator on fake images, `.detach()` stops gradients flowing
back into the Generator, so that step only updates D.

## If something goes wrong on the day

The three output images from a working run are already saved in
`demo/demo_outputs/`, so you can just open those and talk over them.
