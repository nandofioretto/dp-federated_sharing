# Differential Privacy Papers 

## Table of Contents
<a name='_content'></a>

- [Binning Mechanisms](#_binning_mechanism_)

- [Machine Learning](#_machine_learning)
	+ [Theory](#_theory)
	+ [Supervised](#_supervised)
		+ [Classification Decision Trees](#_decision_trees)
		+ [Regression](#_regression)
	+ [Unsupervised](#_unsupervised)
		+ [Clustering](#_clustering) 
		+ [Gaussian](#_Gaussian)

- [Distributed Differential Privacy](#_distributed_differential_privacy)

- [Evaluation Methodology](#_methodology)


## Binning mechanisms
<a name='_binning_mechanism_'></a>

- Towards Accurate Histogram Publication under Differential Privacy
	[[Paper]](https://epubs.siam.org/doi/abs/10.1137/1.9781611973440.68)

	_Xiaojian Zhang Rui Chen Jianliang Xu Xiaofeng Meng Yingtao Xie(SIAM)_

	>_Histograms are the workhorse of data mining and analysis. This paper considers the problem of publishing histograms under differential privacy, one of the strongest privacy models. Existing differentially private histogram publication schemes have shown that clustering (or grouping) is a promising idea to improve the accuracy of sanitized histograms. However, none of them fully exploits the benefit of clustering. In this paper, we introduce a new clustering framework. It features a sophisticated evaluation of the trade-off between the approximation error due to clustering and the Laplace error due to Laplace noise injected, which is normally overlooked in prior work. In particular, we propose three clustering strategies with different orders of run-time complexities. We prove the superiority of our approach by theoretical utility comparisons with the competitors. Our extensive experiments over various standard real-life and synthetic datasets confirm that our technique consistently outperforms existing competitors._

	>[Comments]: _Proposed three different clustering strategies. All of them effectively balance the trade-off between the approximation error and the Laplace error, and therefore achieve better accuracy_

## Machine Learning in Differential Privacy
<a name='_machine_learning'></a>

### Theory
<a name='_theory'></a>

- Differential Privacy and Machine Learning: a Survey and Review 
	[[Paper]](https://arxiv.org/abs/1412.7584)

	_Zhanglong Ji, Zachary C. Lipton, Charles Elkan (Conference)_

	>_The objective of machine learning is to extract useful information from data, while privacy is preserved by concealing information. Thus it seems hard to reconcile these competing interests. However, they frequently must be balanced when mining sensitive data. For example, medical research represents an important application where it is necessary both to extract useful information and protect patient privacy. One way to resolve the conflict is to extract general characteristics of whole populations without disclosing the private information of individuals. 
	In this paper, we consider differential privacy, one of the most popular and powerful definitions of privacy. We explore the interplay between machine learning and differential privacy, namely privacy-preserving machine learning algorithms and learning-based data release mechanisms. We also describe some theoretical results that address what can be learned differentially privately and upper bounds of loss functions for differentially private algorithms. 
	Finally, we present some open questions, including how to incorporate public data, how to deal with missing data in private datasets, and whether, as the number of observed samples grows arbitrarily large, differentially private machine learning algorithms can be achieved at no cost to utility as compared to corresponding non-differentially private algorithms._


### Decision Trees
<a name='_decision_trees'></a>

- A Practical Differentially Private Random Decision Tree Classifier
	[[Paper]](http://www.tdp.cat/issues11/tdp.a082a11.pdf)

	_Geetha Jagannathan, Krishnan Pillaipakkamnatt, Rebecca N Wright (ICDMW 2009)_

	>_In this paper,we study the problem of constructing private classifiers using decision trees, within the framework of differential privacy. We first present experimental evidence that creating a differentially private ID3 tree using differentially private low-level queries does not simultaneously provide good privacy and good accuracy, particularly for small datasets.
	In search of better privacy and accuracy, we then present a differentially private decision tree ensem- ble algorithm based on random decision trees. We demonstrate experimentally that this approach yields good prediction while maintaining good privacy, even for small datasets. We also present dif- ferentially private extensions of our algorithm to two settings: (1) new data is periodically appended to an existing database and (2) the database is horizontally or vertically partitioned between multiple users._

	>[Comments] : Also similar mechanisms


- Differentially Private Data Release for Data Mining
	[[Paper]](http://www.cs.umanitoba.ca/~noman/Papers/MCFY11kdd.pdf)

	_Noman Mohammed, Rui Chen  Benjamin Fung, Philip S Yu (KDD 2011)_

	>_Privacy-preserving data publishing addresses the problem of disclosing sensitive data when mining for useful infor- mation. Among the existing privacy models, ε-differential privacy provides one of the strongest privacy guarantees and has no assumptions about an adversary’s background knowledge. Most of the existing solutions that ensure ε- differential privacy are based on an interactive model, where the data miner is only allowed to pose aggregate queries to the database. In this paper, we propose the first anonymiza- tion algorithm for the non-interactive setting based on the generalization technique. The proposed solution first prob- abilistically generalizes the raw data and then adds noise to guarantee ε-differential privacy. As a sample application, we show that the anonymized data can be used effectively to build a decision tree induction classifier. Experimen- tal results demonstrate that the proposed non-interactive anonymization algorithm is scalable and performs better than the existing solutions for classification analysis._


- Differentially- and non-differentially-private random decision trees
	[[Paper]](https://arxiv.org/pdf/1410.6973.pdf)

	_Mariusz Bojarski, Anna Choromanska, KrzysztofChoromanski, Yann LeCun (2015)_

	>_We consider supervised learning with random decision trees, where the tree construction is completely random. The method is popularly used and works well in practice de- spite the simplicity of the setting, but its statistical mech- anism is not yet well-understood. In this paper we pro- vide strong theoretical guarantees regarding learning with random decision trees. We analyze and compare three dif- ferent variants of the algorithm that have minimal mem- ory requirements: majority voting, threshold averaging and probabilistic averaging. The random structure of the tree enables us to adapt these methods to a differentially-private setting thus we also propose differentially-private versions of all three schemes. We give upper-bounds on the general- ization error and mathematically explain how the accuracy depends on the number of random decision trees. Further- more, we prove that only logarithmic (in the size of the dataset) number of independently selected random decision trees suffice to correctly classify most of the data, even when differential-privacy guarantees must be maintained. We em- pirically show that majority voting and threshold averaging give the best accuracy, also for conservative users requiring high privacy guarantees. Furthermore, we demonstrate that a simple majority voting rule is an especially good candi- date for the differentially-private classifier since it is much less sensitive to the choice of forest parameters than other methods._

- A Differentially Private Decision Forest 
	[[Paper]](http://crpit.com/confpapers/CRPITV168Fletcher.pdf)

	_Sam Fletcher and Md Zahidul Islam (Conference)_

	>_With the ubiquity of data collection in today’s society, protecting each individual’s privacy is a growing con- cern. Differential Privacy provides an enforceable def- inition of privacy that allows data owners to promise each individual that their presence in the dataset will be almost undetectable. Data Mining techniques are often used to discover knowledge in data, however these techniques are not differentially privacy by de- fault. In this paper, we propose a differentially pri- vate decision forest algorithm that takes advantage of a novel theorem for the local sensitivity of the Gini Index. The Gini Index plays an important role in building a decision forest, and the sensitivity of it’s equation dictates how much noise needs to be added to make the forest be differentially private. We prove that the Gini Index can have a substantially lower sensitivity than that used in previous work, leading to superior empirical results. We compare the prediction accuracy of our decision forest to not only previous work, but also to the popular Random Forest algo- rithm to demonstrate how close our differentially pri- vate algorithm can come to a completely non-private forest._

    >[Comments]: A similar technique? 

- Decision Tree Classification with Differential Privacy: A Survey
	[[Paper]](https://arxiv.org/pdf/1611.01919.pdf)

	_Sam Fletcher, Charles Sturt (2017)_

	>_Data mining information about people is becoming increasingly important in the data-driven society of the 21st century. Unfortunately, sometimes there are real-world considerations that conflict with the goals of data mining; sometimes the privacy of the people being data mined needs to be considered. This necessitates that the output of data mining algorithms be modified to preserve privacy while simultaneously not ruining the predictive power of the outputted model. Differential privacy is a strong, enforceable definition of privacy that can be used in data mining algorithms, guaranteeing that nothing will be learned about the people in the data that could not already be discovered without their participation. In this survey, we focus on one particular data mining algorithm – decision trees – and how differential privacy interacts with each of the components that constitute decision tree algorithms. We analyze both greedy and random decision trees, and the conflicts that arise when trying to balance privacy requirements with the accuracy of the model._

	>[Interesting]: _One theme that re-occurred throughout our discussion was the ever-present conflict between privacy and utility. Whenever the sensitive data is queried, there is a cost associated, and that cost needs to be weighed against the benefits. Those benefits depend on the aim of the algorithm; is it aiming to discover knowledge, build a model with high prediction accuracy, both, or something else? These aims each lend themselves to slightly different strategies, changing the cost-benefit analysis that the user makes when deciding when their algorithm should query the data. Depending on how a query is asked, its sensitivity can also change drastically. As we saw in Section 3.3.1, even changing a query slightly from “what are the class counts in this leaf node?” to “what is the majority class label in this leaf node?” allows for a much less noisy answer to be outputted._


- Privacy-preserving Prediction 
	[[Paper]](https://arxiv.org/abs/1803.10266)

	_Cynthia Dwork, Vitaly Feldman (9 May 2018)_

	>_Ensuring differential privacy of models learned from sensitive user data is an important goal that has been studied extensively in recent years. It is now known that for some basic learning problems, especially those involving high-dimensional data, producing an accurate private model requires much more data than learning without privacy. At the same time, in many applications it is not necessary to expose the model itself. Instead users may be allowed to query the prediction model on their inputs only through an appropriate interface. Here we formulate the problem of ensuring privacy of individual predictions and investigate the overheads required to achieve it in several standard models of classification and regression. 
	We first describe a simple baseline approach based on training several models on disjoint subsets of data and using standard private aggregation techniques to predict. We show that this approach has nearly optimal sample complexity for (realizable) PAC learning of any class of Boolean functions. At the same time, without strong assumptions on the data distribution, the aggregation step introduces a substantial overhead. We demonstrate that this overhead can be avoided for the well-studied class of thresholds on a line and for a number of standard settings of convex regression. The analysis of our algorithm for learning thresholds relies crucially on strong generalization guarantees that we establish for all differentially private prediction algorithms._


### Regression
<a name='_regression'></a>
- Differentially Private Significance Tests for Regression Coefficients ((1705.09561))
	[[Paper]](https://arxiv.org/pdf/1705.09561.pdf)

	_Andres F. Barrientos, Jerome P. Reiter, Ashwin Machanavajjhala, Yan Chen (Conference)_

	>_Many data producers seek to provide users access to confidential data without unduly compromising data subjects’ privacy and confidentiality. One general strategy is to require users to do analyses without seeing the confidential data; for example, analysts only get access to synthetic data or query systems that provide disclosure-protected outputs of sta- tistical models. With synthetic data or redacted outputs, the analyst never really knows how much to trust the resulting findings. In particular, if the user did the same analysis on the confidential data, would regression coefficients of interest be statistically significant or not? We present algorithms for assessing this question that satisfy differential privacy. We describe conditions under which the algorithms should give accurate answers about statis- tical significance. We illustrate the properties of the proposed methods using artificial and genuine data._


- Revisiting differentially private linear regression 
	[[Paper]](https://arxiv.org/abs/1803.02596)

	_Yu-Xiang Wang (March 2018)_

	>_We revisit the problem of linear regression under a differential privacy constraint. By consolidating existing pieces in the literature, we clarify the correct dependence of the feature, label and coefficient domain in the optimization error and estimation error, hence revealing the delicate price of differential privacy in statistical estimation and statistical learning. Moreover, we propose simple modifications of two existing DP algorithms: (a) posterior sampling, (b) sufficient statistics perturbation, and show that they can be upgraded into adaptive algorithms that are able to exploit data-dependent quantities and behave nearly optimally for every instance. Extensive experiments are conducted on both simulated data and real data, which conclude that both AdaOPS and AdaSSP outperform the existing techniques on nearly all 36 data sets that we test on._
    
    >[Comments]: 

## Gaussian
<a name='_Gaussian'></a>

- Improving the Gaussian Mechanism for Differential Privacy: Analytical Calibration and Optimal Denoising∗ 
	[[Paper]](https://arxiv.org/abs/1805.06530)

	_Borja Balle and Yu-Xiang Wang (Conference)_

	>_The Gaussian mechanism is an essential building block used in multitude of differentially private data analysis algorithms. In this paper we revisit the Gaussian mechanism and show that the original analysis has several important limitations. Our analysis reveals that the variance formula for the original mechanism is far from tight in the high privacy regime (ε → 0) and it cannot be extended to the low privacy regime (ε → ∞). We address these limitations by developing an optimal Gaussian mechanism whose variance is calibrated directly using the Gaussian cumulative density function instead of a tail bound approximation. We also propose to equip the Gaussian mechanism with a post-processing step based on adaptive estimation techniques by leveraging that the distribution of the perturbation is known. Our experiments show that analytical calibration removes at least a third of the variance of the noise compared to the classical Gaussian mechanism, and that denoising dramatically improves the accuracy of the Gaussian mechanism in the high-dimensional regime._

## Distributed Differential Privacy
<a name='_distributed_differential_privacy'></a>

- Privacy Preserving Analytics on Distributed Medical Data 1806.06477 
	[[Paper]](https://arxiv.org/abs/1806.06477)

	_Marina Blanton, Ah Reum Kang, Subhadeep Karan, Jaroslaw Zola (18 Jun 2018)_

	>_Objective: To enable privacy-preserving learning of high quality generative and discriminative machine learning models from distributed electronic health records.
	Methods and Results: We describe general and scalable strategy to build machine learning models in a provably privacy-preserving way. Compared to the standard approaches using, e.g., differential privacy, our method does not require alteration of the input biomedical data, works with completely or partially distributed datasets, and is resilient as long as the majority of the sites participating in data processing are trusted to not collude. We show how the proposed strategy can be applied on distributed medical records to solve the variables assignment problem, the key task in exact feature selection and Bayesian networks learning.
	Conclusions: Our proposed architecture can be used by health care organizations, spanning providers, insurers, researchers and computational service providers, to build robust and high quality preditive models in cases where distributed data has to be combined without being disclosed, altered or othewise compromised._

    >[Comments]: 


## Evaluation Methodology
<a name='_methodology'></a>

- Comparative Study of Differentially Private Data Synthesis Methods
Categorical data 
Evaluation: The bias, RMSE, 95% coverage probability (CP), and 95% confidence interval (CI) width of π in simulation study 1.
Presentation: Evaluation method over ln(epsilon), with different sample sizes, and different π
Continuous data 
Evaluation: The bias, RMSE, 95% coverage probability (CP), and 95% confidence interval (CI) width of π in simulation study 1.
Presentation: Evaluation method over ln(epsilon), with different sample sizes, and different variance/mean

- 10.1.1.799.7017 
Evaluation method: Accuracy of machine learning software [Weka](https://www.cs.waikato.ac.nz/ml/weka/index.html)
Presentation: Accuracy over the size of the training sample, comparing performance of different algorithms 
Notes: This work poses several future challenges. 
	- The large variance in the experimental results is clearly a problem, and more stable results are desirable even if they come at a cost. One solution might be to consider other stopping rules when splitting nodes, trading possible improvements in accuracy for increased stability. In addition, it may be fruitful to consider different tactics for budget distribution. 
	- Another interesting direction, following the approach presented in [14], is to relax the privacy requirements and allow ruling out rare calculation outcomes that lead to poor results.

- A Differentially Private Decision Forest
Evaluation method: Accuracy of decision forest prediction on 6 UCI datasets, compare to previous studies and benchmark, 10 iterations of stratified 10-fold cross-validation using scikit learn 
Presentation: Accuracy over privacy budget (algorithm that they came up with), comparing performance of different algorithms 

- Differentially Private Model Selection With Penalized and Constrained Likelihood
Evaluation: Accuracy of machine learning over changing model selection via penalized constrained least squares, comparing graphs under different algorithm, and different sample sizes 
Presentation: Accuracy percentage over model selection via penalized constrained least squares
Notes: Larger sample sizes, the faster to reach accuracy. (Doens't necessarily get more accurate)
