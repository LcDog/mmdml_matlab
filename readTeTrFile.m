function [train_text, train_sift, test_text, test_sift, same_pair, dif_pair, train_label, test_label] = readTeTrFile(params)
% ---------------------------------------------------------------------
% USAGE: 
% load train/test data accroding to the path
% ---------------------------------------------------------------------
% DATA FORMAT:
% The dataset contains four files:
%
% 1. train_data.txt
% training data. The first line has the number of training samples.
% Each of the following lines contains one image and the format is 
% [image id] \t [class label] \t [num of tags] \t [num of nonzero sift words] [ a list of tags represented with [tag id]:[tag value] pairs] [ a list of sift words represented with [word id]:[word value] pairs]
% 
% 2. test_data.txt
% testing data. The format is the same as train_data.txt
% 
% 3. train_simi_pair.txt
% similar pairs of training data. The first line contains the number of similar pairs.
% Each of the following lines contains two image ids which are labeled as similar.
% 
% 4. train_diff_pair.txt
% dissimilar pairs of training data. The first line contains the number of dissimilar pairs.
% Each of the following lines contains two image ids which are labeled as dissimilar.
% ----------------------------------------------------------------------
% DEFAULT DATA SOURCE:
% This is the dataset used in the paper Multi-Modal Distance Metric Learning. Please refer to the paper for detailed description.
% 
% If you find the dataset useful, please cite the paper:
% 
% @inproceedings{xie2013multi,
%   title={Multi-modal distance metric learning},
%   author={Xie, Pengtao and Xing, Eric P},
%   booktitle={Proceedings of the Twenty-Third international joint conference on Artificial Intelligence},
%   pages={1806--1812},
%   year={2013},
%   organization={AAAI Press}
% }
% -----------------------------------------------------------------------
% OUT DATA FORMAT:
% train_text 	: (double)train_num * text_dim
% train_sift 	: (double)train_num * sift_dim
% test_text 	: (double)test_num * text_dim
% test_sift 	: (double)test_num * sift_dim
% same_pair 	: (double)same_pair_num * 2
% dif_pair 		: (double)dif_pair_num * 2
% -----------------------------------------------------------------------


	train_file = fopen(params.train_path, 'r');
	test_file = fopen(params.test_path, 'r');
 	if (train_file == -1 || test_file == -1) 
 		error('Wrong Path');
 	end

	[train_num, read_in_num_check] = fscanf(train_file, '%d', 1);
	if (read_in_num_check ~= 1) 
		error('File Format Wrong!');
	end

	[test_num, read_in_num_check] = fscanf(test_file, '%d', 1);
	if (read_in_num_check ~= 1) 
		error('File Format Wrong!');
	end

	% load train data
	all_data_num = train_num + test_num;
	dataID2trainID = zeros(all_data_num, 1);

	train_text = zeros(train_num, params.text_dim);
	train_sift = zeros(train_num, params.sift_dim);
	train_label = zeros(train_num, 1);
	for i = 1:train_num
		read_in_temp = fscanf(train_file, '%d', 4);
		% '+1' for sift/text begin with 0
		dataID2trainID(read_in_temp(1)+1) = i; 
		train_label(i) = read_in_temp(2) + 1;
		for k = 1:read_in_temp(3)
			text_temp = fscanf(train_file, '%d:%d', 2);
			train_text(i, text_temp(1)+1) = text_temp(2);
		end
		for k = 1:read_in_temp(4)
			sift_temp = fscanf(train_file, '%f:%f', 2);
			train_sift(i, sift_temp(1)+1) = sift_temp(2);
		end
	end

	% transfer the same image ID to the train matrix row ID
	same_pair_file = fopen(params.same_train_path, 'r');
	dif_pair_file = fopen(params.dif_train_path, 'r');

	same_pair_num = fscanf(same_pair_file, '%d', 1);
	same_pair = zeros(same_pair_num, 2);
	for i = 1:same_pair_num
		same_pair_temp = fscanf(same_pair_file, '%d %d', 2);
		same_pair(i,1) = dataID2trainID(same_pair_temp(1)+1);
		same_pair(i,2) = dataID2trainID(same_pair_temp(2)+1);
	end

	dif_pair_num = fscanf(dif_pair_file, '%d', 1);
	dif_pair = zeros(dif_pair_num, 2);
	for i = 1:dif_pair_num
		dif_pair_temp = fscanf(dif_pair_file, '%d %d', 2);
		dif_pair(i,1) = dataID2trainID(dif_pair_temp(1)+1);
		dif_pair(i,2) = dataID2trainID(dif_pair_temp(2)+1);
	end

	fclose(same_pair_file);
	fclose(dif_pair_file);

	% load test data
	test_text = zeros(test_num, params.text_dim);
	test_sift = zeros(test_num, params.sift_dim);
	test_label = zeros(test_num, 1);
	for i = 1:test_num
		read_in_temp = fscanf(test_file, '%d', 4);
		test_label(i) = read_in_temp(2) + 1;
		for k = 1:read_in_temp(3)
			text_temp = fscanf(test_file, '%d:%d', 2);
			test_text(i, text_temp(1)+1) = text_temp(2);
		end
		for k = 1:read_in_temp(4)
			sift_temp = fscanf(test_file, '%f:%f', 2);
			test_sift(i, sift_temp(1)+1) = sift_temp(2);
		end
	end


	fclose(train_file);
	fclose(test_file);

end