function [text, sift, t] = Gibbs_sampling (text_, sift_, theta_, eta_, W, U)
% ---------------------------------------------------------------------
% USEAGE:
% conduct Gibbs sampling process defined in Eq.(17)
% ---------------------------------------------------------------------
% INPUT:
% text: text modality; (double)pic_num * text_dim
% sift: sift modality; (double)pic_num * sift_dim
% theta_, eta_, W, U: model parameters for sampling
% ---------------------------------------------------------------------
% OUTPUT: 
% t: E(h) after sampling (pic_num * h_dim)
% text: text modality; (double)pic_num * text_dim
% sift: sift modality; (double)pic_num * sift_dim
% ---------------------------------------------------------------------

	t = get_h(W, U, text_, sift_);

	train_num = size(text_,1);

	if size(theta_,2) == 1
		theta_ = repmat(theta_',train_num, 1);
	else
		theta_ = repmat(theta_,train_num, 1);
	end
	text = 1./(1 + exp(-theta_ - t * W'));

	if size(eta_,2) == 1
		eta_ = repmat(eta_',train_num, 1);
	else
		eta_ = repmat(eta_,train_num, 1);
	end
	sift = eta_ + t * U';

	t = text_ * W + sift_ * U;

end