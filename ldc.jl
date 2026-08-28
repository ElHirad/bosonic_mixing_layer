using ITensors
using LinearAlgebra
using ITensorMPS
using JLD2
using CUDA
using Printf
using Base.Filesystem: basename

CUDA.functional() || error("CUDA is not functional on this node")
CUDA.versioninfo()
CUDA.allowscalar(false)
println("Using GPU: ", CUDA.device())

# Function to extract max_boson from filename
function max_boson_from_filename()
    script = basename(@__FILE__)
    m = match(r"max_boson_(\d+)", script)
    m === nothing && error("Cannot parse max_boson from filename: $script")
    return parse(Int, m.captures[1])
end

# Define bosonic operators according to nmax
function ITensors.space(::SiteType"MyBoson";
                        conserve_qns=true)
  if conserve_qns

    # Define array of pairs of quantum numbers and the dimension of each one
    array = [QN("nb",0)=>1];    # Initialize array of pairs
    for k = 1:nmax
        append!(array, [QN("nb",k)=>1])
    end

    return array
  end

  return nmax+1 # Only return full dimension if no quantum numbers are used
end

function ITensors.op!(Op::ITensor,
                      ::OpName"Num",
                      ::SiteType"MyBoson",
                      s::Index)
    # Fill diagonal
    for k = 1:nmax
        Op[s'=>k+1,s=>k+1] = k
    end

end

function ITensors.op!(Op::ITensor,
                      ::OpName"Num2",
                      ::SiteType"MyBoson",
                      s::Index)
    # Fill diagonal
    for k = 1:nmax
        Op[s'=>k+1,s=>k+1] = k*k
    end

end

function ITensors.op!(Op::ITensor,
                      ::OpName"a",
                      ::SiteType"MyBoson",
                       s::Index)
    # Fill +1 diagonal
    for k = 1:nmax
       Op[s'=>k,s=>k+1] = sqrt(k);
    end

end

function ITensors.op!(Op::ITensor,
                      ::OpName"adag",
                      ::SiteType"MyBoson",
                      s::Index)
    # Fill -1 diagonal
    for k = 1:nmax
       Op[s'=>k+1,s=>k] = sqrt(k);
    end

end

function ITensors.op!(Op::ITensor,
                      ::OpName"Iden",
                      ::SiteType"MyBoson",
                      s::Index)
    # Fill diagonal
    for k = 1:nmax+1
        Op[s'=>k,s=>k] = 1
    end

end

function grid_from_array(k::Integer; n::Integer)
    k < 1 && throw(ArgumentError("k must be ≥ 1"))
    k > n^2 && throw(ArgumentError("k must be ≤ $(n^2)"))

    i = (k - 1) ÷ n + 1           # row (1-based)
    p = (k - 1) % n + 1           # position within the row (1..ncols)

    j = isodd(i) ? p : (n - p + 1)
    return i, j
end;

function array_from_grid(i::Integer, j::Integer; n::Integer)
    (1 ≤ i ≤ n) || throw(ArgumentError("i out of range"))
    (1 ≤ j ≤ n) || throw(ArgumentError("j out of range"))

    p = isodd(i) ? j : (n - j + 1)
    return (i - 1)*n + p
end;

# Define initial state
function InitialState(α_w,α_p,s,N,χ,cutoff)
    n = Int(sqrt(N))  # Number of sites per dimension
    state = [1 for nn=1:2*N] # Initial empty lattice
    ψ0 = MPS(s,state);

    # applying displacement operators:
    for k = 1:N
        i, j = grid_from_array(k; n)
        ψ0 = apply(exp(α_w[i,j]*op("adag",s[k]) - conj(α_w[i,j])*op("a",s[k])), ψ0; cutoff=cutoff, maxdim=χ)
        ψ0 = apply(exp(α_p[i,j]*op("adag",s[k+N]) - conj(α_p[i,j])*op("a",s[k+N])), ψ0; cutoff=cutoff, maxdim=χ)
    end

    return ψ0;
end;

# Calculating expectation values
function ExpVals!(ψ,N,Popul,Norm,timeval) # Here, timeval is the position in the arrays of expectation values where info will be stored

   # The functions expect and correlation_matrix normalize internally the expectation values
   Popul[timeval,:] = real(expect(ψ, "a")); # Population of each site
   Norm[timeval,1] = real(inner(ψ,ψ)); # Norm of the state

end

# Hamiltonian MPO
function Hamiltonians(s,N,δt,Re,δx,xs,ys)

    n = Int(sqrt(N))  # Number of sites per dimension

    # Initiate construction
    ops_w = OpSum()
    ops_p = OpSum()

    # Hamiltonian terms
    for k=1:N

        i, j = grid_from_array(k; n)

        if i != 1 && i != n && j != 1 && j != n  # Only add terms for inner sites, not for boundaries
            w_i₊j = array_from_grid(i+1,j; n)
            w_i₋j = array_from_grid(i-1,j; n)
            w_ij₊ = array_from_grid(i,j+1; n)
            w_ij₋ = array_from_grid(i,j-1; n)

            p_i₊j =  w_i₊j + N
            p_i₋j =  w_i₋j + N
            p_ij₊ =  w_ij₊ + N
            p_ij₋ =  w_ij₋ + N

            # Define factors for single-site operators of Vorticity
            # Convection of Vorticity
            # term -(dp/dy)(dw/dx)
            ops_w -= (1 / (4*(δx^2))),"adag",k,"a",w_i₊j,"a",p_ij₊
            ops_w += (1 / (4*(δx^2))),"adag",k,"a",w_i₊j,"a",p_ij₋
            ops_w += (1 / (4*(δx^2))),"adag",k,"a",w_i₋j,"a",p_ij₊
            ops_w -= (1 / (4*(δx^2))),"adag",k,"a",w_i₋j,"a",p_ij₋
            # term (dp/dx)(dw/dy)
            ops_w += (1 / (4*(δx^2))),"adag",k,"a",w_ij₊,"a",p_i₊j
            ops_w -= (1 / (4*(δx^2))),"adag",k,"a",w_ij₊,"a",p_i₋j
            ops_w -= (1 / (4*(δx^2))),"adag",k,"a",w_ij₋,"a",p_i₊j
            ops_w += (1 / (4*(δx^2))),"adag",k,"a",w_ij₋,"a",p_i₋j

            # Diffusion of Vorticity
            # Diffusion in x
            ops_w += (1 / (Re * δx^2)),"adag",k,"a",w_i₊j
            ops_w += (1 / (Re * δx^2)),"adag",k,"a",w_i₋j
            ops_w += -(2 / (Re * δx^2)),"Num",k
            # Diffusion in y
            ops_w += (1 / (Re * δx^2)),"adag",k,"a",w_ij₊
            ops_w += (1 / (Re * δx^2)),"adag",k,"a",w_ij₋
            ops_w += -(2 / (Re * δx^2)),"Num",k

            # Define factors for single-site operators of Streamfunction
            # Diffusion in x
            ops_p += (1 / (δx^2)),"adag",k+N,"a",p_i₊j
            ops_p += (1 / (δx^2)),"adag",k+N,"a",p_i₋j
            ops_p += -(2 / (δx^2)),"Num",k+N
            # Diffusion in y
            ops_p += (1 / (δx^2)),"adag",k+N,"a",p_ij₊
            ops_p += (1 / (δx^2)),"adag",k+N,"a",p_ij₋
            ops_p += -(2 / (δx^2)),"Num",k+N
            # Adding Vorticity
            ops_p += 40,"adag",k+N,"a",k
        end

    end

    H_w = MPO(ops_w,s);
    H_p = MPO(ops_p,s);

    return H_w, H_p;

end;

function save_mps_checkpoint(ψ_gpu, s, step, tphys;
                             folder="./data/mps_checkpoints",
                             prefix="psi")
    mkpath(folder)

    # Move MPS tensors from GPU memory back to CPU memory
    # ψ_cpu = cpu(ψ_gpu)

    filename = joinpath(
        folder,
        @sprintf("%s_step_%06d_t_%.6f_maxdim_%d.jld2",
                 prefix, step, tphys, maxlinkdim(ψ_gpu))
    )

    jldsave(filename;
        ψ = ψ_gpu,
        sites = s,
        step = step,
        time = tphys,
        maxlinkdim = maxlinkdim(ψ_gpu),
        norm = inner(ψ_gpu, ψ_gpu)
    )

    return filename
end

# Warm-up Code
function warmup()

    N = 16   # Number of sites
    n = Int(sqrt(N))  # Number of sites per dimension

    Re = 10 # Reynolds number
    δx = 1 / (sqrt(N)) # Spatial step

    T = 0.02 # Final time
    δt = 0.01   # Time step
    time = 0.0:δt:T   # Time vector
    cutoff = 1E-6;   # Truncation allowed per step
    χ = 50; # Maximum bond dimension
    tol = 1E-2; # Poisson Solver Tolorence
    maxiter = 1 # Poisson Solver Maximum Iterations
    inter_dt = 1E-4 # Poisson Solver Internal Time Step

    Norm = zeros(1,1); # Norm of evolved state
    Popul = zeros(1,2*N); # Coherent variable per site
    Time_expvals = zeros(1,1); # Time of expectation values

    s = siteinds("MyBoson", 2*N, conserve_qns=false); # For all sites

    xs = range(0, 1 - δx, length=n)
    ys = range(0, 1 - δx, length=n)

    α_w = zeros(n,n)
    α_p = zeros(n,n)
    α_w[:,end] .= (-3 / δx) / 40

    ψ0 = InitialState(α_w,α_p,s,N,χ,cutoff);

    ExpVals!(ψ0, N, Popul, Norm, 1);

    H_w, H_p = Hamiltonians(s,N,δt,Re,δx,xs,ys);

    H_w = cu(H_w);
    H_p = cu(H_p);
    ψ = cu(ψ0);

    for t in 1:length(time)-1

        ψ = tdvp(H_w, δt, ψ; time_step = δt, cutoff = cutoff, maxdim = χ, outputlevel=0, normalize=false);

        # Updating x boundary sites
        for i = 1:n
            # Updating w[:,0]
            k_0 = array_from_grid(i,1; n=n);
            k_1 = array_from_grid(i,2; n=n);
            k_2 = array_from_grid(i,3; n=n);
            w_0 = real(expect(ψ, "a"; sites=k_0));
            p_1 = real(expect(ψ, "a"; sites=k_1));
            p_2 = real(expect(ψ, "a"; sites=k_2));
            w_0_new = ((1 / (δx^2)) * ((-4 * p_1) + (0.5 * p_2))) / 40;
            ψ = apply(cu(exp((w_0_new - w_0)*op("adag",s[k_0]) - conj(w_0_new - w_0)*op("a",s[k_0]))), ψ; cutoff=cutoff, maxdim=χ);
            # Updating w[:,end]
            k_n = array_from_grid(i,n; n=n);
            k_nm1 = array_from_grid(i,n-1; n=n);
            k_nm2 = array_from_grid(i,n-2; n=n);
            w_n = real(expect(ψ, "a"; sites=k_n));
            p_nm1 = real(expect(ψ, "a"; sites=k_nm1));
            p_nm2 = real(expect(ψ, "a"; sites=k_nm2));
            w_n_new = (((1 / (δx^2)) * ((-4 * p_nm1) + (0.5 * p_nm2))) - (3 / δx)) / 40;
            ψ = apply(cu(exp((w_n_new - w_n)*op("adag",s[k_n]) - conj(w_n_new - w_n)*op("a",s[k_n]))), ψ; cutoff=cutoff, maxdim=χ);
        end

        # Updating y boundary sites
        for j = 1:n
            # Updating w[0,:]
            k_0 = array_from_grid(1,j; n=n);
            k_1 = array_from_grid(2,j; n=n);
            k_2 = array_from_grid(3,j; n=n);
            w_0 = real(expect(ψ, "a"; sites=k_0));
            p_1 = real(expect(ψ, "a"; sites=k_1 + N));
            p_2 = real(expect(ψ, "a"; sites=k_2 + N));
            w_0_new = ((1 / (δx^2)) * ((-4 * p_1) + (0.5 * p_2))) / 40;
            ψ = apply(cu(exp((w_0_new - w_0)*op("adag",s[k_0]) - conj(w_0_new - w_0)*op("a",s[k_0]))), ψ; cutoff=cutoff, maxdim=χ);
            # Updating w[end,:]
            k_n = array_from_grid(n,j; n=n);
            k_nm1 = array_from_grid(n-1,j; n=n);
            k_nm2 = array_from_grid(n-2,j; n=n);
            w_n = real(expect(ψ, "a"; sites=k_n));
            p_nm1 = real(expect(ψ, "a"; sites=k_nm1 + N));
            p_nm2 = real(expect(ψ, "a"; sites=k_nm2 + N));
            w_n_new = ((1 / (δx^2)) * ((-4 * p_nm1) + (0.5 * p_nm2))) / 40;
            ψ = apply(cu(exp((w_n_new - w_n)*op("adag",s[k_n]) - conj(w_n_new - w_n)*op("a",s[k_n]))), ψ; cutoff=cutoff, maxdim=χ);
        end

        for inter_n = 1:maxiter
            #ψ_old = ψ;
            ψ = tdvp(H_p, inter_dt, ψ; time_step = inter_dt, cutoff = cutoff, maxdim = χ, outputlevel=0, normalize=false);
            # # Check convergence
            # diff_norm = real(norm(ψ - ψ_old)) / real(norm(ψ_old));
            # if diff_norm < tol
            #     break;
            # end
        end

        fname = save_mps_checkpoint(
                ψ, s, t, t * δt;
                folder="./data/mps_checkpoints",
                prefix="psi_warmup_test"
            )

    end

end

# Main Code
function main()

    N = 256   # Number of sites
    n = Int(sqrt(N))  # Number of sites per dimension

    Re = 100 # Reynolds number
    δx = 1 / (sqrt(N)) # Spatial step

    T = 3.6  # Final time
    δt = 0.002   # Time step
    time = 0.0:δt:T   # Time vector
    tbigstep = 300   # Calculate expectation values each tbigstep times
    num_expvals = Int((length(time)-1)/tbigstep) + 1; # Number of times expectation values will be calculated. The ±1 is to account correctly for t = 0
    cutoff = 1E-20;   # Truncation allowed per step
    χ = 50; # Maximum bond dimension
    tol = 1E-2; # Poisson Solver Tolorence
    maxiter = 2 # Poisson Solver Maximum Iterations
    inter_dt = 1E-4 # Poisson Solver Internal Time Step

    Norm = zeros(num_expvals,1); # Norm of evolved state
    Popul = zeros(num_expvals,2*N); # Coherent variable per site
    Time_expvals = zeros(num_expvals,1); # Time of expectation values

    s = siteinds("MyBoson", 2*N, conserve_qns=false); # For all sites

    xs = range(0, 1 - δx, length=n)
    ys = range(0, 1 - δx, length=n)

    α_w = zeros(n,n)
    α_p = zeros(n,n)
    α_w[:,end] .= (-3 / δx) / 40

    ψ0 = InitialState(α_w,α_p,s,N,χ,cutoff);

    ExpVals!(ψ0, N, Popul, Norm, 1);

    H_w, H_p = Hamiltonians(s,N,δt,Re,δx,xs,ys);

    H_w = cu(H_w);
    H_p = cu(H_p);
    ψ = cu(ψ0);

    count_expvals = 1;

    for t in 1:length(time)-1

        ψ = tdvp(H_w, δt, ψ; time_step = δt, cutoff = cutoff, maxdim = χ, outputlevel=0, normalize=false);

        # Updating x boundary sites
        for i = 1:n
            # Updating w[:,0]
            k_0 = array_from_grid(i,1; n=n);
            k_1 = array_from_grid(i,2; n=n);
            k_2 = array_from_grid(i,3; n=n);
            w_0 = real(expect(ψ, "a"; sites=k_0));
            p_1 = real(expect(ψ, "a"; sites=k_1 + N));
            p_2 = real(expect(ψ, "a"; sites=k_2 + N));
            w_0_new = ((1 / (δx^2)) * ((-4 * p_1) + (0.5 * p_2))) / 40;
            ψ = apply(cu(exp((w_0_new - w_0)*op("adag",s[k_0]) - conj(w_0_new - w_0)*op("a",s[k_0]))), ψ; cutoff=cutoff, maxdim=χ);
            # Updating w[:,end]
            k_n = array_from_grid(i,n; n=n);
            k_nm1 = array_from_grid(i,n-1; n=n);
            k_nm2 = array_from_grid(i,n-2; n=n);
            w_n = real(expect(ψ, "a"; sites=k_n));
            p_nm1 = real(expect(ψ, "a"; sites=k_nm1 + N));
            p_nm2 = real(expect(ψ, "a"; sites=k_nm2 + N));
            w_n_new = (((1 / (δx^2)) * ((-4 * p_nm1) + (0.5 * p_nm2))) - (3 / δx)) / 40;
            ψ = apply(cu(exp((w_n_new - w_n)*op("adag",s[k_n]) - conj(w_n_new - w_n)*op("a",s[k_n]))), ψ; cutoff=cutoff, maxdim=χ);
        end

        # Updating y boundary sites
        for j = 1:n
            # Updating w[0,:]
            k_0 = array_from_grid(1,j; n=n);
            k_1 = array_from_grid(2,j; n=n);
            k_2 = array_from_grid(3,j; n=n);
            w_0 = real(expect(ψ, "a"; sites=k_0));
            p_1 = real(expect(ψ, "a"; sites=k_1 + N));
            p_2 = real(expect(ψ, "a"; sites=k_2 + N));
            w_0_new = ((1 / (δx^2)) * ((-4 * p_1) + (0.5 * p_2))) / 40;
            ψ = apply(cu(exp((w_0_new - w_0)*op("adag",s[k_0]) - conj(w_0_new - w_0)*op("a",s[k_0]))), ψ; cutoff=cutoff, maxdim=χ);
            # Updating w[end,:]
            k_n = array_from_grid(n,j; n=n);
            k_nm1 = array_from_grid(n-1,j; n=n);
            k_nm2 = array_from_grid(n-2,j; n=n);
            w_n = real(expect(ψ, "a"; sites=k_n));
            p_nm1 = real(expect(ψ, "a"; sites=k_nm1 + N));
            p_nm2 = real(expect(ψ, "a"; sites=k_nm2 + N));
            w_n_new = ((1 / (δx^2)) * ((-4 * p_nm1) + (0.5 * p_nm2))) / 40;
            ψ = apply(cu(exp((w_n_new - w_n)*op("adag",s[k_n]) - conj(w_n_new - w_n)*op("a",s[k_n]))), ψ; cutoff=cutoff, maxdim=χ);
        end

        for inter_n = 1:maxiter
            ψ_old = ψ;
            ψ = tdvp(H_p, 50 * inter_dt, ψ; time_step = inter_dt, cutoff = cutoff, maxdim = χ, outputlevel=0, normalize=false);
            # # Check convergence
            diff_norm = real(norm(ψ - ψ_old)) / max(real(norm(ψ_old)), 1e-15)
            println("Difference norm: $diff_norm")
            CUDA.reclaim()
            println("Memory freed after TDVP step: ", CUDA.memory_status())
            if diff_norm < tol
                break;
            end
        end

        #normalize!(ψ)

        if(mod(t,tbigstep)== 0)

            println("Calculating expectation values for $(t) number of steps")
            count_expvals = count_expvals + 1;
            Time_expvals[count_expvals] = t*δt;
            # The state is not normalized, just the expectation values in the function
            ExpVals!(ψ, N, Popul, Norm, count_expvals);
            println("Maximum bond dimension at this step: $(maxlinkdim(ψ))")

            # Save full MPS checkpoint
            fname = save_mps_checkpoint(
                ψ, s, t, t * δt;
                folder="./data/mps_checkpoints",
                prefix="psi_max_boson_$(max_boson)_gpu_cutoff1e20_dt_0_001"
            )

            println("Saved MPS checkpoint: $fname")

            @info "GPU mem" free=CUDA.available_memory() total=CUDA.total_memory()
            CUDA.memory_status()

        end

    end

    @save "./data/Popul_max_boson_$(max_boson)_gpu_cutoff1e20_dt_0_001.jld2" Popul
    @save "./data/Norm_max_boson_$(max_boson)_gpu_cutoff1e20_dt_0_001.jld2" Norm

end

max_boson = max_boson_from_filename()
nmax = 2; # Maximum number of bosons per site
warmup()
println("Warm-up done. Starting main code.")
nmax = max_boson; # Maximum number of bosons per site
main()