## Requirements

To install all the requirements

```bash
pip install -r requirements.txt
```

## Example of command to generate archive images

* Generate all the archive images of format `png` and `svg`
* from files in `example/`
* save them in `images/`
* show titles on the figures
* only consider the files `proj_{n}.dat` to create the images where `n` is:
    * a multiple of `100`
    * inferior or equal to `10000`

```bash
python -m generate_figures --format png svg --path example/ --save images/ --show-titles --gen-step 100 --gen-max 10000
```