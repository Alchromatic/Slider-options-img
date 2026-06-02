
In the submitted closure package, the M7.1 logic is primarily in:

scripts/run_measured_pairwise_unmix.py
src/measured_pairwise_model.py
src/color_metrics.py

But to simplify things, I consolidated the current M4/M7.1 logic into a single Python entry file: m7_1_unified.py

This file includes the M4 route logic, the M7.1 measured-pairwise prediction model, and the M7.1 unmix/search flow. The CSV/JSON files remain separate because they are measured Trycolors UI data inputs, not separate code versions.

In terms of versions: M4 is the current forward-router candidate, while M7.1 is the latest inverse/unmix closure candidate. The older M5/M6/M6.1 files were intermediate research candidates and should not be treated as separate production versions.

The branch history previously created several Python files, and I agree it can be difficult to keep them straight.

To clarify, the older Python files are mostly historical/research branches, not six separate production versions. 

The current recommended setup is:

forward_version = “m4”
unmix_version = “m7_1”

The reason I separate them is that not every version has the same input/output type. 
- Some versions are forward-mix models, where the input is a recipe and the output is a predicted mixed color. 
- Other versions are inverse/unmix models, where the input is a target color and the output is a proposed recipe.

So for the API, I would structure it as:

forward_mix(recipe, version=“m4”)
unmix(target_hex, version=“m7_1”)

The packages in this folder is meant to consolidate the version history and show how the different versions can be exposed through a versioned API. 

The current unified file is:
current_unified/m7_1_unified.py

The older Python files are included for historical comparison/reference, but the current recommended branch is M4 for forward mixing and M7.1 for inverse/unmixing.

So the previous submissions do not need to be deleted, but they should be treated as historical. 

This latest consolidated package should be used as the main reference going forward.