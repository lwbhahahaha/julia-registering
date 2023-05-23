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
	# Pkg.add("Plots")
	# Pkg.add("ProgressBars")
	# Pkg.add("ImageFiltering")
	# Pkg.add(url = "https://github.com/Dale-Black/DistanceTransforms.jl", rev= "master")
	# Pkg.add("ImageEdgeDetection")
end

# ╔═╡ aa653711-5656-4e20-90c4-378e4bd71735
begin
	# using RegisterMismatch, RegisterQD # cpu
	using RegisterMismatchCuda, RegisterQD # gpu
	using DICOM
	using CUDA
	using Plots
	using EasyFit
	using Dates
	using Images, ImageView
	using Statistics
	using ProgressBars
	using Printf
	using DistanceTransforms
	using ImageFiltering
	using ImageEdgeDetection
	using CoordinateTransformations
end

# ╔═╡ f229b2c5-3605-424e-b077-3173ef9b026e
begin
	function set_pixels_to_zero(arr_in, t1, t2; pixel_min = 0)
		arr = copy(arr_in)
	    Threads.@threads for i in eachindex(arr)
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
	
	sharpen_kernel = [0.0 -1.0 0.0; -1.0 5.0 -1.0; 0.0 -1.0 0.0]	
	Laplacian_kernel = [-1.0 -1.0 -1.0; -1.0 8.0 -1.0; -1.0 -1.0 -1.0]

	function crop_by_radius!(img; r_2 = 64150, min_pixel = -2048)
		Threads.@threads for i in CartesianIndices(img)
			(i[1] - 256)^2 + (i[2] - 256)^2 > r_2 && (img[i] = min_pixel)
		end
	end
end;

# ╔═╡ 49964431-0c4e-41bc-93f9-6d9ac804b665
"""
	This functions finds values for BB(Boundary Box) in this format: 
		[up, down, left, right]
"""
function find_BB(curr_limb_dicom, l; offset = 50)
	curr_limb_dicom == nothing && return
	up, down, left, right = nothing, nothing, nothing, nothing
	# top to buttom
	for x = 1 : 512
		up==nothing || break
		for slice_idx = 1 : l
			Threads.@threads for y = 1 : 512
				curr_limb_dicom[slice_idx, x, y] == -1024 || (up=x;break)
			end
			up==nothing || break
		end
	end
	# buttom to top
	for x = 512 : -1 : 1
		down==nothing || break
		for slice_idx = 1 : l
			Threads.@threads for y = 1 : 512
				curr_limb_dicom[slice_idx, x, y] == -1024 || (down=x;break)
			end
			down==nothing || break
		end
	end
	# left to right
	for y = 1 : 512
		left==nothing || break
		for slice_idx = 1 : l
			Threads.@threads for x = 1 : 512
				curr_limb_dicom[slice_idx, x, y] == -1024 || (left=y;break)
			end
			left==nothing || break
		end
	end
	# right to left
	for y = 512 : -1 : 1
		right==nothing || break
		for slice_idx = 1 : l
			Threads.@threads for x = 1 : 512
				curr_limb_dicom[slice_idx, x, y] == -1024 || (right=y;break)
			end
			right==nothing || break
		end
	end
	return [max(1, up-offset), min(512, down+offset), max(1, left-offset), min(512, right+offset)]
end

# ╔═╡ cc06cbab-d8eb-4d82-ad58-974de746d0f3
"""
	This function reads all DICOM pixel values given paths.
"""
function read_DICOM_pixel_values(found_limb_dcm, path_to_v1_folder, matched_file_names, path_to_DICOM_folder, path_to_Limb_dcm_folder, l)
	curr_v1_dicom = Array{Int16, 3}(undef, l, 512, 512)
	curr_v2_dicom = Array{Int16, 3}(undef, l, 512, 512)
	curr_limb_dicom = found_limb_dcm ? Array{Int16, 3}(undef, l, 512, 512) : nothing
	Threads.@threads for i = 1 : l
		v1_dicom_path = joinpath(path_to_v1_folder, matched_file_names[i])
		v2_dicom_path = joinpath(joinpath(path_to_DICOM_folder, "02/", matched_file_names[i]))
		limb_dicom_path = joinpath(path_to_Limb_dcm_folder, matched_file_names[l-i+1])
		curr_v1_dicom[i, :, :] = dcm_parse(v1_dicom_path)[(0x7fe0, 0x0010)]
		curr_v2_dicom[i, :, :] = dcm_parse(v2_dicom_path)[(0x7fe0, 0x0010)]
		if found_limb_dcm
			curr_limb_dicom[i, :, :] = dcm_parse(limb_dicom_path)[(0x7fe0, 0x0010)]	
		end
	end
	return curr_v1_dicom, curr_v2_dicom, curr_limb_dicom
end

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

			found_limb_dcm = false
			path_to_Limb_dcm_folder = joinpath("input/", study_name, "Segment_dcm/Limb_dcm/")
			
			path_to_v1_folder = joinpath(path_to_DICOM_folder, "01/")
			matched_file_names = []
			for f in readdir(path_to_v1_folder)
				if isfile(joinpath(path_to_DICOM_folder, "02/", f))
					ct += 1
					push!(matched_file_names, f)
					if !found_limb_dcm && isfile(joinpath(path_to_Limb_dcm_folder, f))
						found_limb_dcm = true
					end
				end
			end
			l = size(matched_file_names)[1]

			# Read all DICOM images
			curr_v1_dicom, curr_v2_dicom, curr_limb_dicom = read_DICOM_pixel_values(found_limb_dcm, path_to_v1_folder, matched_file_names, path_to_DICOM_folder, path_to_Limb_dcm_folder, l)
			# Get BB is limb_dcm is found
			BB = find_BB(curr_limb_dicom, l)
			
			push!(result, (study_name, "", matched_file_names, curr_v1_dicom, curr_v2_dicom, BB))
			
			print("\t\tFound $(size(matched_file_names)[1]) slices.")
			if found_limb_dcm
				println(" Also found limb_dcm for this study.")
			else
				println(" Limb_dcm for this study is not found.")
			end
		else
			# 2 or more ACQs
			ACQs = readdir(joinpath("input/", study_name))
			println("\t\t$(size(ACQs)[1]) acquisitions.")
			for acq in ACQs
				path_to_DICOM_folder = joinpath("input/", study_name, acq, "DICOM/")
				path_to_Limb_dcm_folder = joinpath("input/", study_name, acq, "Segment_dcm/Limb_dcm/")

				found_limb_dcm = false
				
				path_to_v1_folder = joinpath(path_to_DICOM_folder, "01/")
				matched_file_names = []
				for f in readdir(path_to_v1_folder)
					if isfile(joinpath(path_to_DICOM_folder, "02/", f))
						ct += 1
						push!(matched_file_names, f)
						if !found_limb_dcm && isfile(joinpath(path_to_Limb_dcm_folder, f))
							found_limb_dcm = true
						end
					end
				end
				l = size(matched_file_names)[1]

				# Read all DICOM images
				curr_v1_dicom, curr_v2_dicom, curr_limb_dicom = read_DICOM_pixel_values(found_limb_dcm, path_to_v1_folder, matched_file_names, path_to_DICOM_folder, path_to_Limb_dcm_folder, l)
				# Get BB is limb_dcm is found
				BB = find_BB(curr_limb_dicom, l)
				
				push!(result, (study_name, acq, matched_file_names, curr_v1_dicom, curr_v2_dicom, BB))

				printstyled("\t\t$acq"; color = :yellow, bold = true)
				print(": Found $(size(matched_file_names)[1]) slices.")
				if found_limb_dcm
					println(" Also found limb_dcm for this study.")
				else
					println(" Limb_dcm for this study is not found.")
				end
			end
			path_to_DICOM_folder = joinpath("input/", study_name, "DICOM/")
		end
		println()
	end
	println("Total $ct DICOM slices.\nDone!\n")
	return result
end

# ╔═╡ a8b3d5fb-b945-4bbf-8cf5-f09af105cd34
"""
	This functions preprocesses the input iamge.
"""
function preprocess_image(img, crop, BB; medi = 5, blur = 2, debug = false)
	img_out = copy(img)
	# binarylize
		bgd = img .< 500
		img_out[bgd] .= -2048
		# img_out[.!bgd] .= 1
	# # sharpen
		img_out2 = imfilter(img, sharpen_kernel)
	# median
		img_out2 = mapwindow(median!, img_out2, (medi, medi))
	# sharpen
		img_out2 = imfilter(img_out2, sharpen_kernel);
	# # laplacian
	# 	# img_out = imfilter(img_out, Laplacian_kernel)
	# 	img_out2 = imfilter(img_out2, Kernel.Laplacian())
	# edge detection
		# imshow(normalize(img_out2); name="step 5")
		detect_edges!(img_out2, Canny(spatial_scale = 1.4, low = ImageEdgeDetection.Percentile(90), high = ImageEdgeDetection.Percentile(99)))
		# imshow(normalize(img_out2); name="step 6")
	# crop
		crop_by_radius!(img_out2; r_2 = 55000, min_pixel = 0)
	# largest connected component
		img_out3 = dilate(img_out2, [1,2])
		img_out3 = label_components(img_out3, [1,2])
		component_to_keep = argmax(component_lengths(img_out3)[2:end])
		img_out3 = (img_out3 .!= component_to_keep)
		img_out2[img_out3] .= 0.0
		img_out2 = dilate(img_out2, [1,2])
		# imshow(normalize(img_out2); name="step 7")
		for (i, v) in enumerate(img_out2)
			if v==1 && img_out[i] == -2048
				img_out[i] = 1250
			end
		end
		# img_out = img_out .+ img_out2 .* 3000
	# crop
		crop_by_radius!(img_out; r_2 = 55000, min_pixel = -2048)
		# imshow(normalize(img_out); name="step 4")
	# # DT 
	# 	img_out = transform(img_out, Maurer())
	# 	img_out = maximum(img_out) .- img_out
	# # crop
	# 	crop_by_radius!(img_out; min_pixel = 0.0)
	# crop
	if crop && BB!=nothing
		up, down, left, right = BB
		img = img[up:down, left:right]
		img_out = img_out[up:down, left:right]
	end
	return img, img_out
end

# ╔═╡ d2eb0488-b1f0-4b15-aa3c-0c4c9fb01138
"""
	This function applies RegisterQD(https://github.com/HolyLab/RegisterQD.jl) to images.

		mode = 1(qd_translate): register images by shifting one with respect to another (translations only); (~90 seconds for 821 slice on gpu)
	
		mode = 2(qd_rigid): register images using rotations and translations; (~12 minutes for 821 slices on gpu)
	
		mode = 3(qd_affine): register images using arbitrary affine transformations. (~2.1 hours for 821 slices on gpu)
"""
function Apply_RegisterQD(pair_images, crop; mode = 3, num_threads = 2, debug = false, debug_num_slices=60)
	println("Applying QuadDIRECT algorithm...")
	if mode == 1
		num_threads = 6
	end
	results = []
	errors = []
	for (study_name, ACQ_name, dicom_file_names, curr_v1_dicom, curr_v2_dicom, BB) in pair_images
		
		printstyled("\t$study_name"; color = :yellow, bold = true)
		if ACQ_name!= ""
			printstyled(", $ACQ_name"; color = :yellow, bold = true)
		end
		println(":\n")
		
		num_slices = size(dicom_file_names)[1]
		# num_slices = 100

		tforms = Array{Any}(undef, num_slices)
		mms = Array{Float64}(undef, num_slices)
		curr_rslt = Array{Any}(undef, num_slices)
		prev_tform = nothing 
		# for j = 1:num_slices
		for i = 1:num_threads: num_slices
			Threads.@threads for j = i : i+num_threads-1
				if j <= num_slices


					dicom_file = dicom_file_names[j]
					_moving = curr_v1_dicom[j, :, :]
					# println(size(_moving))
					_fixed = curr_v2_dicom[j, :, :]
					# println(size(_moving))
					
					_moving, moving = preprocess_image(_moving, crop, BB)
					_fixed, fixed = preprocess_image(_fixed, crop, BB) 
					
					# Call RegisterQD.jl
					if mode == 1
						if prev_tform == nothing
							tforms[j], mms[j] = qd_translate(fixed, moving, (10,10))
						else
							tforms[j], mms[j] = qd_translate(fixed, moving, (10,10); initial_tfm = prev_tform)
						end
					elseif mode == 2
						if prev_tform == nothing
							tforms[j], mms[j] = qd_rigid(fixed, moving, (10,10), 0.2)
						else
							tforms[j], mms[j] = qd_rigid(fixed, moving, (10,10), 0.2; initial_tfm = prev_tform)
						end
					else
						if prev_tform == nothing
							tforms[j], mms[j] = qd_affine(fixed, moving, (10,10))
						else
							tforms[j], mms[j] = qd_affine(fixed, moving, (10,10); initial_tfm = prev_tform)
						end
					end
					
					output_dir = joinpath("output/", study_name, ACQ_name, "DICOM/01_corrected/")
					output_dir_v2 = joinpath("output/", study_name, ACQ_name, "DICOM/02/")
					v1_dicom_path = joinpath("input/", study_name, ACQ_name, "DICOM/01", dicom_file)
					v2_dicom_path = joinpath("input/", study_name, ACQ_name, "DICOM/02", dicom_file)
					curr_rslt[j] = (_moving, _fixed, output_dir, v1_dicom_path, v2_dicom_path, output_dir_v2, dicom_file)
					j == i+num_threads-1 && (prev_tform = tforms[j])

					
				end
			end
		end
		push!(results, (curr_rslt, tforms, mms))
	end
	println("Done!")
	return results, errors
end

# ╔═╡ ae7392b0-17d7-459e-94d2-4e1e694675b1
# pair_images = find_pair_images();

# ╔═╡ 828073ec-7d60-40f7-9fbb-30f8e53de0b1
# ╠═╡ disabled = true
#=╠═╡
begin
	(study_name, ACQ_name, dicom_file_names, curr_v1_dicom, curr_v2_dicom, BB) = pair_images[1]

	j = 103

	dicom_file = dicom_file_names[j]
	_moving = curr_v1_dicom[j, :, :]
	_fixed = curr_v2_dicom[j, :, :]

	medi = 5
	blur = 2
	crop = false
	
	img = _moving
	img_out = copy(img)
	# binarylize
		bgd = img .< 500
		img_out[bgd] .= -2048
		# img_out[.!bgd] .= 1
	# # sharpen
		img_out2 = imfilter(img, sharpen_kernel)
	# median
		img_out2 = mapwindow(median!, img_out2, (medi, medi))
	# sharpen
		img_out2 = imfilter(img_out2, sharpen_kernel);
	# # laplacian
	# 	# img_out = imfilter(img_out, Laplacian_kernel)
	# 	img_out2 = imfilter(img_out2, Kernel.Laplacian())
	# edge detection
		# imshow(normalize(img_out2); name="step 5")
		detect_edges!(img_out2, Canny(spatial_scale = 1.4, low = ImageEdgeDetection.Percentile(90), high = ImageEdgeDetection.Percentile(99)))
		# imshow(normalize(img_out2); name="step 6")
	# crop
		crop_by_radius!(img_out2; r_2 = 55000, min_pixel = 0)
	# largest connected component
		img_out3 = dilate(img_out2, [1,2])
		img_out3 = label_components(img_out3, [1,2])
		# imshow(img_out3; name="step 5.2")
		component_to_keep = argmax(component_lengths(img_out3)[2:end])
	
		# println(component_to_keep)
	
		img_out3 = (img_out3 .!= component_to_keep)
		img_out2[img_out3] .= 0.0
		img_out2 = dilate(img_out2, [1,2])
		# imshow(normalize(img_out2); name="step 7")
		for (i, v) in enumerate(img_out2)
			if v==1 && img_out[i] == -2048
				img_out[i] = 1250
			end
		end
		# img_out = img_out .+ img_out2 .* 3000
	# crop
		crop_by_radius!(img_out; r_2 = 55000, min_pixel = -2048)
		# imshow(normalize(img_out); name="step 4")
	# # DT 
	# 	img_out = transform(img_out, Maurer())
	# 	img_out = maximum(img_out) .- img_out
	# # crop
	# 	crop_by_radius!(img_out; min_pixel = 0.0)
	# crop
	if crop && BB!=nothing
		up, down, left, right = BB
		img = img[up:down, left:right]
		img_out = img_out[up:down, left:right]
	end
	# imshow(img_out; name="step 1")
	_moving = img
	# imshow(_fixed; name="_moving")
	moving = img_out

	img = copy(_fixed)
	img_out = copy(img)
	# binarylize
		bgd = img .< 500
		img_out[bgd] .= -2048
		# img_out[.!bgd] .= 1
	# # sharpen
		img_out2 = imfilter(img, sharpen_kernel)
	# median
		img_out2 = mapwindow(median!, img_out2, (medi, medi))
	# sharpen
		img_out2 = imfilter(img_out2, sharpen_kernel);
	# # laplacian
	# 	# img_out = imfilter(img_out, Laplacian_kernel)
	# 	img_out2 = imfilter(img_out2, Kernel.Laplacian())
	# edge detection
		# imshow(normalize(img_out2); name="step 5")
		detect_edges!(img_out2, Canny(spatial_scale = 1.4, low = ImageEdgeDetection.Percentile(90), high = ImageEdgeDetection.Percentile(99)))
		# imshow(normalize(img_out2); name="step 6")
	# crop
		crop_by_radius!(img_out2; r_2 = 55000, min_pixel = 0)
		# imshow(normalize(img_out2); name="step 6.1")
	# largest connected component
		img_out3 = dilate(img_out2, [1,2])
		img_out3 = label_components(img_out3, [1,2])
		# imshow(img_out3; name="step 6.2")
		component_to_keep = argmax(component_lengths(img_out3)[2:end])
		# println(component_to_keep)
	
		img_out3 = (img_out3 .!= component_to_keep)
		# imshow(normalize(img_out3); name="step 6.3")
		img_out2[img_out3] .= 0.0
		img_out2 = dilate(img_out2, [1,2])
		# imshow(normalize(img_out2); name="step 7")
		for (i, v) in enumerate(img_out2)
			if v==1 && img_out[i] == -2048
				img_out[i] = 1250
			end
		end
		# img_out = img_out .+ img_out2 .* 3000
	# crop
		crop_by_radius!(img_out; r_2 = 55000, min_pixel = -2048)
		# imshow(img_out; name="step 8")
	# # DT 
	# 	img_out = transform(img_out, Maurer())
	# 	img_out = maximum(img_out) .- img_out
	# # crop
	if crop && BB!=nothing
		up, down, left, right = BB
		img = img[up:down, left:right]
		img_out = img_out[up:down, left:right]
	end
	# imshow(img_out; name="step 2")
	_fixed = img
	fixed = img_out
	
	tform, mm = qd_affine(fixed, moving, (10,10))
	corrected_v1 = warp(_moving, tform, axes(_fixed))

	
	
	imshow(colorview(RGB, normalize(_fixed), normalize(_moving), zeroarray); name="with motion $j");
	# imshow(colorview(RGB, normalize(_fixed), normalize(fixed), zeroarray); name="fixed with edge $j");
	# imshow(colorview(RGB, normalize(moving), normalize(_moving), zeroarray); name="moving with edge $j");
	# imshow(fixed; name="fixed $j");
	# imshow(moving; name="moving $j");
	imshow(colorview(RGB, normalize(_fixed), normalize(corrected_v1), zeroarray); name="registered $j");
	
end
  ╠═╡ =#

# ╔═╡ 76fc8cbb-8fde-4c93-9446-1e006f1edd03
# imshow(colorview(RGB, normalize(img_out), normalize(img), zeroarray); name="registered $j");

# ╔═╡ 6e4334c0-7fc6-4c58-b6cb-ec02f120a51d
# let
# 	k_size = 55
# 	j = 1
# 	temp = Array{Float64}(undef, l)
# 	temp2 = Array{Float64}(undef, l)
# 	Threads.@threads for i in x
# 		temp[i] = tfromss[i].linear[j]
# 		temp2[i] = tfromss_copy[i].linear[j]
# 	end
# 	temp2 = mapwindow(median!, temp2, (k_size))
# 	plot(x, [temp, temp2])
# end

# ╔═╡ 1a765b7b-6713-430f-bf5b-8e11529772d1
"""
	This function deals with the file system and saves the corrected images.
"""
function save_images(results; debug = false, debug_num_slices = 60, 
		k_size = 35)
	
	println("\nSaving images...")
	for (rslt, tforms, mms) in results
		# pass median filter through tforms
		temptemp = []
		tforms_copy = []
		l = size(tforms)[1]
		x = collect(1:l)
		for j = 1:4
			temp = Array{Float64}(undef, l)
			Threads.@threads for i in x
				temp[i] = tforms[i].linear[j]
			end
			temp = mapwindow(median!, temp, (k_size))
			push!(temptemp, temp)
		end
		for j = 1:2
			temp = Array{Float64}(undef, l)
			Threads.@threads for i in x
				temp[i] = tforms[i].translation[j]
			end
			temp = mapwindow(median!, temp, (k_size))
			push!(temptemp, temp)
		end
		for i in x
			push!(tforms_copy, AffineMap([temptemp[1][i] temptemp[3][i]; temptemp[2][i] temptemp[4][i]], [temptemp[5][i], temptemp[6][i]]))
		end
		# Apply tfrom and save
		Threads.@threads for i = 1: l
			_moving, _fixed, output_dir, v1_dicom_path, v2_dicom_path, output_dir_v2, dicom_file = rslt[i]
			corrected_v1 = warp(_moving, tforms_copy[i], axes(_fixed))
			
			moving = dcm_parse(v1_dicom_path)
			moving[(0x7fe0, 0x0010)] = round.(Int16, corrected_v1)
			moving[(0x0028, 0x0010)] = size(corrected_v1)[1]
			moving[(0x0028, 0x0011)] = size(corrected_v1)[2]
			isdir(output_dir) || (mkpath(output_dir))
			dcm_write(joinpath(output_dir, dicom_file), moving)
			
			fixed = dcm_parse(v2_dicom_path)
			fixed[(0x7fe0, 0x0010)] = round.(Int16, _fixed)
			fixed[(0x0028, 0x0010)] = size(_fixed)[1]
			fixed[(0x0028, 0x0011)] = size(_fixed)[2]
			isdir(output_dir_v2) || (mkpath(output_dir_v2))
			dcm_write(joinpath(output_dir_v2, dicom_file), fixed)
		end
		return mms
	end
	println("Done!")
end

# ╔═╡ 89dc624f-79e1-41fe-a70b-f1888d654bc8
"""
	This function works as the main function which wraps all other functions in this notebook.

	This function takes two inputs: mode(int) and crop(bool)
		mode = 1,2 or 3
		crop = true or false
"""
function run(mode, crop; debug = false, debug_num_slices=60)
	pair_images = find_pair_images();
	results, errors = Apply_RegisterQD(pair_images, crop; mode = mode, debug = debug, debug_num_slices=debug_num_slices);
	# save errors
	open("errors.txt","a") do io
		println(io, "================================================")
		println(io, Dates.format(now(), "HH:MM") )
		for str in errors
	   		println(io, str)
		end
	end
	return save_images(results; debug = debug, debug_num_slices = debug_num_slices)
end

# ╔═╡ 92667783-ad86-4e3c-92de-6bf92830ded6
# ╠═╡ show_logs = false
mms = run(3, true);

# ╔═╡ 66ec4a32-75d3-433a-b26c-2479300222e0
# plot(collect(1:size(mms)[1]), mms)

# ╔═╡ ea4ccff9-485d-4c4e-a336-58fb32e8b7b1
# let
# 	m = maximum(mms)
# 	for i = 1 : 821
# 		mms[i] == m  && (println(i))
# 	end
# end

# ╔═╡ 51741da3-f547-404d-8091-022cc34ecca5
# let
# 	i = 10
# 	_moving, _fixed, output_dir, v1_dicom_path, dicom_file = rslt[i]
# 	corrected_v1 = warp(_moving, tforms_copy[i], axes(_fixed))

# 	imshow(colorview(RGB, normalize(_fixed), normalize(corrected_v1), zeroarray); name="registered $i");
# end

# ╔═╡ 3144b6a9-d188-4edb-bd37-43a67aafd097
# begin
# 	pair_images = find_pair_images()
# 	slice_idx = 326
# 	v1_dicom_path = joinpath("input/", pair_images[1][1], pair_images[1][2], "DICOM/01/", pair_images[1][3][slice_idx])
# 	v2_dicom_path = joinpath("input/", pair_images[1][1], pair_images[1][2], "DICOM/02/", pair_images[1][3][slice_idx])
# 	_moving = dcm_parse(v1_dicom_path)[(0x7fe0, 0x0010)]
# 	_fixed = dcm_parse(v2_dicom_path)[(0x7fe0, 0x0010)]
# 	imshow(colorview(RGB, normalize(_fixed), normalize(_moving), zeroarray); name="original")
# 	moving = transform(preprocess_image(_moving), Maurer())
# 	fixed = transform(preprocess_image(_fixed), Maurer())
# 	tform, mm = qd_rigid(fixed, moving, (10,10), 0.1)
# 	corrected_v1 = warp(_moving, tform, axes(_fixed))
# 	imshow(colorview(RGB, normalize(_fixed), normalize(corrected_v1), zeroarray); name="registered")
# end

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
# ╟─49964431-0c4e-41bc-93f9-6d9ac804b665
# ╟─cc06cbab-d8eb-4d82-ad58-974de746d0f3
# ╟─41b24195-1b85-4780-9864-2765b9068ed2
# ╟─a8b3d5fb-b945-4bbf-8cf5-f09af105cd34
# ╠═d2eb0488-b1f0-4b15-aa3c-0c4c9fb01138
# ╠═ae7392b0-17d7-459e-94d2-4e1e694675b1
# ╟─828073ec-7d60-40f7-9fbb-30f8e53de0b1
# ╠═76fc8cbb-8fde-4c93-9446-1e006f1edd03
# ╠═6e4334c0-7fc6-4c58-b6cb-ec02f120a51d
# ╟─1a765b7b-6713-430f-bf5b-8e11529772d1
# ╟─89dc624f-79e1-41fe-a70b-f1888d654bc8
# ╠═92667783-ad86-4e3c-92de-6bf92830ded6
# ╠═66ec4a32-75d3-433a-b26c-2479300222e0
# ╠═ea4ccff9-485d-4c4e-a336-58fb32e8b7b1
# ╠═51741da3-f547-404d-8091-022cc34ecca5
# ╟─3144b6a9-d188-4edb-bd37-43a67aafd097
# ╟─9eb4b50d-a2a4-435e-8d91-8cef67441810
# ╟─00f6699a-7245-4b54-94b5-5d6bc43b4fae
# ╟─8cdc9a3c-0f16-4cff-8d22-cc5ca76bc667
# ╟─7a4ff5a1-9b86-445f-b2de-6331b01b6825
# ╟─5916defc-2154-45ae-912f-179583278e3b
# ╟─f933e04b-a7a1-4dc9-bec8-b60a929eeeeb
# ╟─05a48358-9e4f-4da0-912a-a0c84f9fb96d
