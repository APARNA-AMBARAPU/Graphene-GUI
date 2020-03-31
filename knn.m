%% K Nearest Neighbors (KNN)

% 1,2,3,4 columns i.e X0Y1,Defect,Strain rate, Temperature of the rawdata.csv are the parameters
X = [rawdata{:,1},rawdata{:,2},rawdata{:,3},rawdata{:,4}];

% column 5 i.e Fracture strain as the target-1
Y1 = rawdata{:,5};
% column 6 i.e Fracture strength as the target-2
Y2 = rawdata{:,6};
% column 5 i.e Young's Modulus as the target-3
Y3 = rawdata{:,7};

%random seed
rng(10);

% fitting knn using in-bulit function fitcknn
Model1 = fitcknn(X,Y1,'NumNeighbors',5);
Model2 = fitcknn(X,Y2,'NumNeighbors',5);
Model3 = fitcknn(X,Y3,'NumNeighbors',5);

% Loss computed for the Data
rloss1 = resubLoss(Model1);
disp('Loss Model1');
disp(rloss1);
rloss2 = resubLoss(Model2);
disp('Loss Model2');
disp(rloss2);
rloss3 = resubLoss(Model3);
disp('Loss Model3');
disp(rloss3);

% Cross Validation models
CV_model1 = crossval(Model1);
CV_model2 = crossval(Model2);
CV_model3 = crossval(Model3);

% k-fold cross validation error
kloss1 = kfoldLoss(CV_model1);
disp('k-fold loss CV_model1');
disp(kloss1);
kloss2 = kfoldLoss(CV_model2);
disp('k-fold loss CV_model2');
disp(kloss2);
kloss3 = kfoldLoss(CV_model3);
disp('k-fold loss CV_model3');
disp(kloss3);

o1 = predict(Model1,[0,0,0.001,100]);
disp('Fracture strain');
disp(o1);
o2 = predict(Model2,[0,0,0.001,100]);
disp('Fracture strength');
disp(o2);
o3 = predict(Model3,[0,0,0.001,100]);
disp("Young's Modulus");
disp(o3);
