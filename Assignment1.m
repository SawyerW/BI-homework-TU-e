% Load data and transform into array
load Assignment1.mat
d_array = table2array(D);
 
fprintf('Data loaded \nSize of target data: %d (%f) \n',sum(table2array(D(:,4))),sum(table2array(D(:,4)))/height(D))
fprintf('Size of non-target data: %d (%f)\n',height(D)-sum(table2array(D(:,4))),(height(D)-sum(table2array(D(:,4))))/height(D))
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Question 1:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
% Create deviding index with 50% Training, 25% Testing and 25% Validation
[trainInd,valInd,testInd] = dividerand(height(D),0.5,0.25,0.25);
 
% Divide data using the index
training_data = d_array(trainInd,:);
test_data = d_array(testInd,:);
validation_data = d_array(valInd,:);
 
fprintf('Size of training data: %d\n',length(training_data))
fprintf('Size of test data: %d\n',length(test_data))
fprintf('Size of validation data: %d\n',length(validation_data))
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Question 2:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Training decision tree')
% Train the decision tree model with parameter MinParent 20
tree=fitctree(training_data(:,1:3),training_data(:,4),'MinParent',20);
% View the trained decision tree
view(tree,'mode','graph');
 
 
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Question 3:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
% Try different MinParent range from 1 to 1500 with step 10
mps = 1:10:1501;
% Store evaluation result (F1) on different MinParent
F1s = [];
precisions = [];
recalls = [];
max_mp = -1;
max_f1 = -1;
 
% Try different value of mp to see evaluation result
for mp = mps
    % Train the tree using training data
    tree_new=fitctree(training_data(:,1:3),training_data(:,4),'MinParent',mp);
    % Predict label of validation data using the tree generated
    pred_val=predict(tree_new,validation_data(:,1:3));
    actual_val=validation_data(:,4);
 
    % Calculate precision, recall and F1 value
    true_positive = sum((pred_val == 1) & (actual_val == 1));
    false_positive = sum((pred_val == 1) & (actual_val == 0));
    false_negative = sum((pred_val == 0) & (actual_val == 1));
    precision = true_positive / (true_positive + false_positive);
    recall = true_positive / (true_positive + false_negative);
    F1 = 2 * precision * recall / (precision + recall);
    F1s = [F1s F1];
    precisions = [precisions precision];
    recalls = [recalls recall];
    fprintf('MinParent:%d, precision:%f, recall:%f, F1:%f \n',mp,precision, recall, F1)
     
    % Save the current maximum F1 and corresponding Min Parent value
    if(F1>max_f1)
        max_f1 = F1;
        max_mp = mp;
    end
     
end
 
% Plot the performance curve
plot(mps,F1s,mps,precisions,'--',mps,recalls,'--')
 
% Add labels, title and legend
xlabel('Min Parent')
ylabel('Metrics')
title('Performance Evaluation over MinParent Parameter')
legend('F1 value','Precision','Recall')
 
% Point out maximum F1 and corresponding Min Parent value in the plot
text(max_mp,max_f1,strcat('\leftarrow Min Parent=',num2str(max_mp),' F1=',num2str(max_f1)))
 
% Evaluate performance on test dataset using the Min Parent value we got
 
 
% Train the tree using training data
tree_final=fitctree(training_data(:,1:3),training_data(:,4),'MinParent',max_mp);
% Predict label of test data using the tree generated
pred_val=predict(tree_final,test_data(:,1:3));
actual_val=test_data(:,4);
 
% Calculate precision, recall and F1 value
true_positive = sum((pred_val == 1) & (actual_val == 1));
false_positive = sum((pred_val == 1) & (actual_val == 0));
false_negative = sum((pred_val == 0) & (actual_val == 1));
precision = true_positive / (true_positive + false_positive);
recall = true_positive / (true_positive + false_negative);
F1 = 2 * precision * recall / (precision + recall);
 
% Print the test evaluation result
fprintf('Test data evaluation: MinParent:%d, precision:%f, recall:%f, F1:%f \n',max_mp,precision, recall, F1)
% View the final decision tree
view(tree_final,'mode','graph');
