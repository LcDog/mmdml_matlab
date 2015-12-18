function t = get_h(W, U, text, sift)
% ---------------------------------------------------------------------
% USEAGE:
% get_h: calculate E(h) for each image
% ---------------------------------------------------------------------
% INPUT:
% W: coupling matrix defined in the paper (text_dim * h_dim)
% U: coupling matrix defined in the paper (sift_dim * h_dim)
% text: text modality; (double)pic_num * text_dim
% sift: sift modality; (double)pic_num * sift_dim
% ---------------------------------------------------------------------
% OUTPUT: 
% t: E(h), the mathematical expectation of hidden topics, 
% 		pic_num * h_dim
% ---------------------------------------------------------------------

	h_dim = size(W, 2);
	pic_num = size(text, 1);

	t = zeros(pic_num, h_dim);
	for k = 1:pic_num
		for i = 1:h_dim
			t(k,i) = text(k,:) * W(:,i) + sift(k,:) * U(:,i);
		end
	end

end