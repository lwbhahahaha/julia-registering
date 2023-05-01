### A Pluto.jl notebook ###
# v0.19.14

using Markdown
using InteractiveUtils

# ╔═╡ 88c6543c-a03f-4529-b7de-4bfbf09d1fd5
begin
	using Pkg
	Pkg.activate(".")
	# Pkg.add(url = "https://github.com/HolyLab/RegisterMismatch.jl")
	# Pkg.add("RegisterQD")
	# Pkg.add("DICOM")
	# Pkg.add("CUDA")
	# Pkg.add("Dates")
	# Pkg.add("ImageView")
	# Pkg.add("Images")
	# Pkg.add("ProgressBars")
end

# ╔═╡ aa653711-5656-4e20-90c4-378e4bd71735
begin
	using RegisterMismatch, RegisterQD # cpu
	# using RegisterMismatchCuda, RegisterQD # gpu
	using DICOM
	using CUDA
	using Dates
	using Images, ImageView
	using Statistics
	using ProgressBars
	using Printf
end

# ╔═╡ f229b2c5-3605-424e-b077-3173ef9b026e
begin
	function set_pixels_to_zero(arr_in, t1, t2; pixel_min = 0)
		arr = copy(arr_in)
	    for i in eachindex(arr)
	        t1 < arr[i] < t2 || (arr[i] = pixel_min)
	    end
	    return arr
	end
	
	function normalize(pic)
		rslt = copy(pic)
		a, b = minimum(rslt), maximum(rslt)
		rslt = (rslt .- a) ./ (b-a)
		return rslt
	end
	
end;

# ╔═╡ 41b24195-1b85-4780-9864-2765b9068ed2
"""
	This function deals with the file system and locate the dicom files.
	return format:
	[
		(study_name, ACQ_name, dicom_files),
		(study_name, ACQ_name, dicom_files),
		...
	]
"""
function find_pair_images()
	ct = 0
	println("Locating pair images...\n")
	result = []
	study_names = readdir("input/")
	for (i,study_name) in enumerate(study_names)
		print("\t$i. Study found: ")
		printstyled("$study_name\n"; color = :yellow, bold = true)
		path_to_DICOM_folder = joinpath("input/", study_name, "DICOM/")
		if isdir(path_to_DICOM_folder)
			# Only 1 ACQ
			println("\t\t1 acquisition.")

			path_to_v1_folder = joinpath(path_to_DICOM_folder, "01/")
			matched_file_names = []
			for f in readdir(path_to_v1_folder)
				if isfile(joinpath(path_to_DICOM_folder, "02/", f))
					ct += 1
					push!(matched_file_names, f)
				end
			end
			
			push!(result, (study_name, "", matched_file_names))
			
			println("\t\tFound $(size(matched_file_names)[1]) slices.")
		else
			# 2 or more ACQs
			ACQs = readdir(joinpath("input/", study_name))
			println("\t\t$(size(ACQs)[1]) acquisitions.")
			for acq in ACQs
				path_to_v1_folder = joinpath("input/", study_name, acq, "DICOM/01/")
				matched_file_names = []
				for f in readdir(path_to_v1_folder)
					if isfile(joinpath("input/", study_name, acq, "DICOM/02/", f))
						ct += 1
						push!(matched_file_names, f)
					end
				end

				push!(result, (study_name, acq, matched_file_names))

				printstyled("\t\t$acq"; color = :yellow, bold = true)
				println(": Found $(size(matched_file_names)[1]) slices.")
			end
			path_to_DICOM_folder = joinpath("input/", study_name, "DICOM/")
		end
		println()
	end
	println("Total $ct DICOM slices.\nDone!\n")
	return result
end

# ╔═╡ d2eb0488-b1f0-4b15-aa3c-0c4c9fb01138
"""
	This function applies RegisterQD(https://github.com/HolyLab/RegisterQD.jl) to images.

		mode = 1(qd_translate): register images by shifting one with respect to another (translations only); (~90 seconds for 821 slice on gpu)
	
		mode = 2(qd_rigid): register images using rotations and translations; (~12 minutes for 821 slices on gpu)
	
		mode = 3(qd_affine): register images using arbitrary affine transformations. (~2.1 hours for 821 slices on gpu)
"""
function Apply_RegisterQD(pair_images; mode = 3, num_threads = 3, debug = false, debug_num_slices=60)
	println("Applying QuadDIRECT algorithm...")
	if mode == 1
		num_threads = 6
	end
	results = []
	errors = []
	for (study_name, ACQ_name, dicom_file_names) in pair_images
		red_flag = false
		printstyled("\t$study_name"; color = :yellow, bold = true)
		if ACQ_name!= ""
			printstyled(", $ACQ_name"; color = :yellow, bold = true)
		end
		println(":\n")
		num_slices = size(dicom_file_names)[1]
		debug && (num_slices = debug_num_slices)
		curr_rslt = Array{Any}(undef, num_slices)
		prev_tform = nothing 
		for i = 1:num_threads: num_slices
			Threads.@threads for j = i : i+num_threads-1
				if j <= num_slices
					dicom_file = dicom_file_names[j]
					v1_dicom_path = joinpath("input/", study_name, ACQ_name, "DICOM/01/", dicom_file)
					v2_dicom_path = joinpath("input/", study_name, ACQ_name, "DICOM/02/", dicom_file)
					if !(isfile(v1_dicom_path))
						printstyled("ERROR(File not found): $(v1_dicom_path)"; color = :red, bold = true)
					else
						# Read DICOM
						_moving, _fixed = nothing, nothing
						try
							_moving = dcm_parse(v1_dicom_path)[(0x7fe0, 0x0010)]
						catch e
							push!(errors, v1_dicom_path)
							red_flag = true
							break
						end
						try
							_fixed = dcm_parse(v2_dicom_path)[(0x7fe0, 0x0010)]
						catch e
							push!(errors, v2_dicom_path)
							red_flag = true
							break
						end
						moving = set_pixels_to_zero(_moving, 100, 450; pixel_min = -2048)
						fixed = set_pixels_to_zero(_fixed, 100, 450; pixel_min = -2048)
						
						# Call RegisterQD.jl

						tform = nothing
						if mode == 1
							if prev_tform == nothing
								tform, mm = qd_translate(fixed, moving, (10,10))
							else
								tform, mm = qd_translate(fixed, moving, (10,10); initial_tfm = prev_tform)
							end
						elseif mode == 2
							if prev_tform == nothing
								tform, mm = qd_rigid(fixed, moving, (10,10), 0.1)
							else
								tform, mm = qd_rigid(fixed, moving, (10,10), 0.1; initial_tfm = prev_tform)
							end
						else
							if prev_tform == nothing
								tform, mm = qd_affine(fixed, moving, (10,10))
							else
								tform, mm = qd_affine(fixed, moving, (10,10); initial_tfm = prev_tform)
							end
						end
						
						corrected_v1 = warp(_moving, tform, axes(fixed))
						output_dir = joinpath("output/", study_name, ACQ_name, "DICOM/01_corrected/")
						curr_rslt[j] = (v1_dicom_path, v2_dicom_path, output_dir, dicom_file, corrected_v1)
						j == i+num_threads-1 && (prev_tform = tform)
					end
				end
			end
			red_flag && (break)
		end
		red_flag || (push!(results, curr_rslt))
	end
	println("Done!")
	return results, errors
end

# ╔═╡ e4acb706-23c3-4d78-8d78-f1cc42fc10f8
"""
	3D Version of Apply_RegisterQD()
"""
function Apply_RegisterQD_3D(pair_images; mode = 3)
	println("Applying QuadDIRECT algorithm...")
	results = []
	errors = []
	for (study_name, ACQ_name, dicom_file_names) in pair_images
		red_flag = false
		printstyled("\t$study_name"; color = :yellow, bold = true)
		if ACQ_name!= ""
			printstyled(", $ACQ_name"; color = :yellow, bold = true)
		end
		println(":\n")
		num_slices = size(dicom_file_names)[1]
		
		# Read DICOM
		_moving_3D = Array{Int16,3}(undef, num_slices, 512, 512)
		_fixed_3D = Array{Int16,3}(undef, num_slices, 512, 512)
		Threads.@threads for slice_idx = 1 : num_slices
			v1_dicom_path = joinpath("input/", study_name, ACQ_name, "DICOM/01/", dicom_file_names[slice_idx])
			v2_dicom_path = joinpath("input/", study_name, ACQ_name, "DICOM/02/", dicom_file_names[slice_idx])
			try
			_moving_3D[slice_idx, :, :] = dcm_parse(v1_dicom_path)[(0x7fe0, 0x0010)]
			catch e
					push!(errors, v1_dicom_path)
					red_flag = true
					break
			end
			try
			_fixed_3D[slice_idx, :, :] = dcm_parse(v2_dicom_path)[(0x7fe0, 0x0010)]
			catch e
					push!(errors, v2_dicom_path)
					red_flag = true
					break
			end
		end
		red_flag && (continue)

		# Fitler images
		moving = set_pixels_to_zero(_moving_3D, 100, 450; pixel_min = -2048)
		fixed = set_pixels_to_zero(_fixed_3D, 100, 450; pixel_min = -2048)
		println(size(moving))
		println(size(fixed))
		# Call RegisterQD.jl
		tform = nothing
		if mode == 1
			tform, mm = qd_translate(fixed, moving, (15,15,15))
		elseif mode == 2
			tform, mm = qd_rigid(fixed, moving, (15,15,15), 0.1)
		else
			tform, mm = qd_affine(fixed, moving, (15,15,15))
		end
		corrected_v1 = warp(_moving_3D, tform, axes(fixed))
		
		# Save
		push!(results, (
		joinpath("input/", study_name, ACQ_name, "DICOM/01/"), 
		joinpath("input/", study_name, ACQ_name, "DICOM/02/"),
		joinpath("output/", study_name, ACQ_name, "DICOM/01_corrected/"), 
		dicom_file_names, corrected_v1))
	end
	println("Done!")
	return results, errors
end

# ╔═╡ 1a765b7b-6713-430f-bf5b-8e11529772d1
"""
	This function deals with the file system and saves the corrected images.
"""
function save_images(results; debug = false, debug_num_slices = 60)
	println("\nSaving images...")
	for rslt in results
		num_slices = debug ? debug_num_slices : size(rslt)[1]
		Threads.@threads for i = 1: num_slices
			v1_dicom_path, v2_dicom_path, output_dir, dicom_file, corrected_v1 = rslt[i]
			moving = dcm_parse(v1_dicom_path)
			moving[(0x7fe0, 0x0010)] = round.(Int16, corrected_v1)
			isdir(output_dir) || (mkpath(output_dir))
			dcm_write(joinpath(output_dir, dicom_file), moving)
		end
	end
	println("Done!")
end

# ╔═╡ 392f3e42-bfb4-4d0e-91ce-dba78faa7e52
"""
	3D version of save_images().
"""
function save_images_3D(results)
	println("\nSaving images...")
	for rslt in results
		v1_dir, v2_dir, out_dir, file_names, corrected_v1 = rslt
		num_slices = size(file_names)[1]
		Threads.@threads for i = 1: num_slices
			output = dcm_parse(joinpath(v1_dir, file_names[i]))
			output[(0x7fe0, 0x0010)] = round.(Int16, corrected_v1[i, :, :])
			isdir(out_dir) || (mkpath(out_dir))
			dcm_write(joinpath(out_dir, file_names[i]), output)
		end
	end
	println("Done!")
end

# ╔═╡ 89dc624f-79e1-41fe-a70b-f1888d654bc8
"""
	This function works as the main function which wraps all other functions in this notebook.
"""
function run(mode; debug = false, debug_num_slices=60)
	pair_images = find_pair_images();
	results, errors = Apply_RegisterQD(pair_images; mode = mode, debug = debug, debug_num_slices=debug_num_slices);
	# save errors
	open("errors.txt","a") do io
		println(io, "================================================")
		println(io, Dates.format(now(), "HH:MM") )
		for str in errors
	   		println(io, str)
		end
	end
	save_images(results; debug = debug, debug_num_slices = debug_num_slices)
end

# ╔═╡ 236c68ac-52ad-46a3-9ca5-1d69242f4fc1
"""
	3D version of run().
"""
function run_3D(mode; debug = false, debug_num_slices=60)
	pair_images = find_pair_images();
	results, errors = Apply_RegisterQD_3D(pair_images; mode = mode);
	# save errors
	open("errors.txt","a") do io
		println(io, "================================================")
		println(io, Dates.format(now(), "HH:MM") )
		for str in errors
	   		println(io, str)
		end
	end
	save_images_3D(results)
end

# ╔═╡ 92667783-ad86-4e3c-92de-6bf92830ded6
run_3D(1)

# ╔═╡ 9eb4b50d-a2a4-435e-8d91-8cef67441810
# results_mode1 = Apply_RegisterQD(pair_images; mode = 1, debug = true);

# ╔═╡ 00f6699a-7245-4b54-94b5-5d6bc43b4fae
# results_mode2 = Apply_RegisterQD(pair_images; mode = 2, debug = true);

# ╔═╡ 8cdc9a3c-0f16-4cff-8d22-cc5ca76bc667
# results_mode3 = Apply_RegisterQD(pair_images; mode = 3, debug = true);

# ╔═╡ 7a4ff5a1-9b86-445f-b2de-6331b01b6825
# idx = 10

# ╔═╡ 5916defc-2154-45ae-912f-179583278e3b
# let
# 	results = results_mode1
# 	# Check result
# 	v1_dicom_path, v2_dicom_path, output_path, corrected_v1 = results[1][idx]
	
# 	moving = dcm_parse(v1_dicom_path)[(0x7fe0, 0x0010)]
# 	fixed = dcm_parse(v2_dicom_path)[(0x7fe0, 0x0010)]

# 	imshow(colorview(RGB, normalize(fixed), normalize(moving), zeroarray); name="original")
# 	imshow(normalize(corrected_v1 - moving); name="diff mode=1");
# 	imshow(colorview(RGB, normalize(fixed), normalize(corrected_v1), zeroarray); name="registered mode=1");
# end

# ╔═╡ f933e04b-a7a1-4dc9-bec8-b60a929eeeeb
# let
# 	results = results_mode2
# 	# Check result
# 	v1_dicom_path, v2_dicom_path, output_path, corrected_v1 = results[1][idx]
	
# 	moving = dcm_parse(v1_dicom_path)[(0x7fe0, 0x0010)]
# 	fixed = dcm_parse(v2_dicom_path)[(0x7fe0, 0x0010)]

# 	# imshow(colorview(RGB, normalize(fixed), normalize(moving), zeroarray); name="original mode=2")
# 	imshow(normalize(corrected_v1 - moving); name="diff mode=2");
# 	imshow(colorview(RGB, normalize(fixed), normalize(corrected_v1), zeroarray); name="registered mode=2");
# end

# ╔═╡ 05a48358-9e4f-4da0-912a-a0c84f9fb96d
# let
# 	results = results_mode3
# 	# Check result
# 	v1_dicom_path, v2_dicom_path, output_path, corrected_v1 = results[1][idx]
	
# 	moving = dcm_parse(v1_dicom_path)[(0x7fe0, 0x0010)]
# 	fixed = dcm_parse(v2_dicom_path)[(0x7fe0, 0x0010)]

# 	# imshow(colorview(RGB, normalize(fixed), normalize(moving), zeroarray); name="original mode=3")
# 	imshow(normalize(corrected_v1 - moving); name="diff mode=3");
# 	imshow(colorview(RGB, normalize(fixed), normalize(corrected_v1), zeroarray); name="registered mode=3");
# end

# ╔═╡ Cell order:
# ╠═88c6543c-a03f-4529-b7de-4bfbf09d1fd5
# ╠═aa653711-5656-4e20-90c4-378e4bd71735
# ╟─f229b2c5-3605-424e-b077-3173ef9b026e
# ╟─41b24195-1b85-4780-9864-2765b9068ed2
# ╟─d2eb0488-b1f0-4b15-aa3c-0c4c9fb01138
# ╟─e4acb706-23c3-4d78-8d78-f1cc42fc10f8
# ╟─1a765b7b-6713-430f-bf5b-8e11529772d1
# ╠═392f3e42-bfb4-4d0e-91ce-dba78faa7e52
# ╟─89dc624f-79e1-41fe-a70b-f1888d654bc8
# ╠═236c68ac-52ad-46a3-9ca5-1d69242f4fc1
# ╠═92667783-ad86-4e3c-92de-6bf92830ded6
# ╠═9eb4b50d-a2a4-435e-8d91-8cef67441810
# ╠═00f6699a-7245-4b54-94b5-5d6bc43b4fae
# ╠═8cdc9a3c-0f16-4cff-8d22-cc5ca76bc667
# ╠═7a4ff5a1-9b86-445f-b2de-6331b01b6825
# ╠═5916defc-2154-45ae-912f-179583278e3b
# ╠═f933e04b-a7a1-4dc9-bec8-b60a929eeeeb
# ╠═05a48358-9e4f-4da0-912a-a0c84f9fb96d
