This is the dataset used in the paper Multi-Modal Distance Metric Learning. Please refer to the paper for detailed description.

If you find the dataset useful, please cite the paper:

@inproceedings{xie2013multi,
  title={Multi-modal distance metric learning},
  author={Xie, Pengtao and Xing, Eric P},
  booktitle={Proceedings of the Twenty-Third international joint conference on Artificial Intelligence},
  pages={1806--1812},
  year={2013},
  organization={AAAI Press}
}

The dataset contains four files:

1. train_data.txt

training data. The first line has the number of training samples.
Each of the following lines contains one image and the format is 

[image id] \t [class label] \t [num of tags] \t [num of nonzero sift words] [ a list of tags represented with [tag id]:[tag value] pairs] [ a list of sift words represented with [word id]:[word value] pairs]

2. test_data.txt

testing data. The format is the same as train_data.txt

3. train_simi_pair.txt

similar pairs of training data. The first line contains the number of similar pairs.
Each of the following lines contains two image ids which are labeled as similar.

4. train_diff_pair.txt

dissimilar pairs of training data. The first line contains the number of dissimilar pairs.
Each of the following lines contains two image ids which are labeled as dissimilar.

