function [theta_, eta_, W, U] = minimize_step(theta_init, eta_init, W_init, U_init, ...
											  train_text, train_sift, same_pair, dif_pair, params)
% ---------------------------------------------------------------------
% USEAGE: 
% minimize the object function according to params
% ---------------------------------------------------------------------
% INPUT:
% theta_init, eta_init, W_init, U_init: initial data
% params: parameters of the algorithm
% ---------------------------------------------------------------------
% OUTPUT:
% theta_, eta_, W, U : value after minimization step
% ---------------------------------------------------------------------

	step_size = params.iter_step_size;

	theta_ 	= theta_init;
	eta_ 	= eta_init;
	W 		= W_init;
	U 		= U_init;

	text_step = train_text;
	sift_step = train_sift;

	for i = 1:params.max_iters
		[theta_grad, eta_grad, W_grad, U_grad] = get_grad (text_step, sift_step, same_pair, dif_pair, ...
														   theta_, eta_, W, U, params);

		theta_ 	= theta_ - step_size * theta_grad;
		eta_ 	= eta_ - step_size * eta_grad;
		W 		= W - step_size * W_grad;
		U 		= U - step_size * U_grad;

		[text_step, sift_step, t_step] = Gibbs_sampling (text_step, sift_step, theta_, eta_, W, U);
		if mod(i,5) == 0
			i
			sum(sum(W))
		end
	end

end