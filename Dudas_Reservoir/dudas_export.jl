# Exports the saved results of Dudas et al. (2023), "Quantum reservoir computing
# implementation on coherently coupled quantum oscillators" (npj Quantum Information 9, 64),
# from their JLD2 files to plain CSV files in dudas_data/, so Python can read them.
# What it needs and how to run it: see the notes at the end of this file.

using JLD2, DelimitedFiles

root = joinpath(@__DIR__, "..", "Research_Set")   # their fig3-fig7 folders  [note 1]
out = joinpath(@__DIR__, "dudas_data")
mkpath(out)

sinesquare = "half_sine_offset_0.6_sin_square_features_eA=1.2e6_eB=1.2e6_coupling=700.0MHz_kappas=17.0_21.0MHz_mesmax=5_sampling=1.jld2"
dephasing = "dephasing_sin_square_features_eA=1.2e6_eB=1.2e6_coupling=700.0MHz_kappas=17.0_21.0MHz_mesmax=5_sampling=1.jld2"
files = [("fig3", sinesquare, "fig3"),                  # sine/square reservoir measurements
         ("fig4", dephasing, "fig4_dephasing"),         # the same with photon-number dephasing (100 MHz)
         ("fig7", "fig7data_coupling.jld2", "fig7"),    # Mackey-Glass error vs delay, g = 700 MHz and g = 0
         ("fig7", "fig7data_kappa.jld2", "fig7")]       # Mackey-Glass error vs delay, two dissipation rates

for (dir, file, tag) in files
    data = load(joinpath(root, dir, file))
    for (name, value) in data
        value isa AbstractArray{<:Real} || continue
        writedlm(joinpath(out, "$(tag)_$(name).csv"), value, ',')
        println("$(tag)_$(name).csv  ", size(value))
    end
end


# ========================================================================================
# EXPLANATION
# ========================================================================================
#
# [note 0] About this script
#
# QCML102426.py and Dudas_Python_Weak.py check their own simulation against the CSVs this
# script writes. The CSVs are already in dudas_data/, so this only needs rerunning if they go.
#
# Needs only the JLD2 package. Add it once (offline works) and then run:
#   julia -e 'using Pkg; Pkg.offline(true); Pkg.add("JLD2")'
#   julia dudas_export.jl
#
# [note 1] Where the JLD2 files come from
#
# The JLD2 files are the saved results that Dudas et al. published with their code, in its
# folders fig3, fig4 and fig7. They are not part of this repository. To regenerate
# dudas_data/, place those folders in a folder named Research_Set next to this repository's
# folders (or change `root` to point at them).
