function s = SetDefaultParams(s)
% --------------------------------------------------------------
% USEAGE:
% Sets default parameters
% --------------------------------------------------------------
% INPUT:
% s: user-specified parameters that are used instead of defaults
% --------------------------------------------------------------
% USE in scripts
% if (~exist('params','var')),
%     params = struct();
% end
% params = SetDefaultParams(params);
% --------------------------------------------------------------

if (isfield(s, 'train_path') == 0),
    if (isunix())
    	s.train_path = './data_ijcai13_mmdl/train_data.txt';
    else
    	s.train_path = '.\data_ijcai13_mmdl\train_data.txt';
    end    		
end

if (isfield(s, 'test_path') == 0),
    if (isunix())
    	s.test_path = './data_ijcai13_mmdl/test_data.txt';
    else
    	s.test_path = '.\data_ijcai13_mmdl\test_data.txt';   
    end
end

if (isfield(s, 'same_train_path') == 0),
    if (isunix())
    	s.same_train_path = './data_ijcai13_mmdl/train_simi_pairs.txt';
    else
    	s.same_train_path = '.\data_ijcai13_mmdl\train_simi_pairs.txt';   
    end
end

if (isfield(s, 'dif_train_path') == 0),
    if (isunix())
    	s.dif_train_path = './data_ijcai13_mmdl/train_diff_pairs.txt';
    else
    	s.dif_train_path = '.\data_ijcai13_mmdl\train_diff_pairs.txt'; 
    end
end

if (isfield(s, 'text_dim') == 0),
    s.text_dim = 1000;
end

if (isfield(s, 'sift_dim') == 0),
    s.sift_dim = 1024;
end

if (isfield(s, 'latent_dim') == 0),
    s.latent_dim = 100;
end

if (isfield(s, 'lamda_1') == 0),
   s.lamda_1 = 0.1; 
end

if (isfield(s, 'lamda_2') == 0),
   s.lamda_2 = 100; 
end

if (isfield(s, 'iter_step_size') == 0),
    s.iter_step_size = 1e-6;
end

if (isfield(s, 'max_iters') == 0),
    s.max_iters = 200;
end

if (isfield(s, 'cd_max_iter') == 0),
   s.cd_max_iter = 1; 
end

if (isfield(s, 'cd_convergence') == 0),
   s.cd_convergence = 1e-3; 
end

% if (isfield(s, 'em_max_iter') == 0),
%    s.em_max_iter = 20; 
% end

% if (isfield(s, 'em_convergence') == 0),
%    s.em_convergence = 1e-5; 
% end
