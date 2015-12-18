function [theta_init, eta_init, W_init, U_init] = initdata(train_text, train_sift, params)
% ---------------------------------------------------------------------
% USEAGE:
% get initial data
% ---------------------------------------------------------------------
% INPUT:
% text: text modality; (double)pic_num * text_dim
% sift: sift modality; (double)pic_num * sift_dim
% ---------------------------------------------------------------------
% OUTPUT: 
% theta_init, eta_init, W_init, U_init: init data
% ---------------------------------------------------------------------
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% remain to be realized by SVD
% !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% ---------------------------------------------------------------------

	theta_init = zeros(params.text_dim, 1);
	eta_init = zeros(params.sift_dim, 1);
	W_init = ones(params.text_dim, params.latent_dim) ./ 95;
	U_init = ones(params.sift_dim, params.latent_dim) ./ 95;

end