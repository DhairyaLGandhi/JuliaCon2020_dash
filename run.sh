wget https://julialang-s3.julialang.org/bin/linux/x64/1.4/julia-1.4.2-linux-x86_64.tar.gz

tar -xvf julia-1.4.2-linux-x86_64.tar.gz
./julia-1.4.2/bin/julia --project utrain.jl $PORT
