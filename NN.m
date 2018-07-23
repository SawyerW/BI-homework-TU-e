%trainno = randi([1,180],1,120)
%testno = randi(([1,180] - trainno),1, 30)
%data_selected = term(:,2:22);
data_selected = term(:,[3,4,9,11,13,14,15,16,17,18,19,22]);
new_train = table2array(data_selected);
data_X = transpose(new_train);
data_target =term(:,23);
new_target = table2array(data_target);
target_data = transpose(new_target);
%target_0 = target_data;
%target_1 = 1-target_data;
%target_2 =[target_0;target_1];
%target_0_test=target_0(:,21118:25517);
temp = 0;
mx_f = 0;
my_f = 0;
%m = [1:50];
for mx = 1:50
     
    for my = 1:50
             net=patternnet([mx,7]);
             net.divideFcn = 'divideind';
             net.divideParam.trainInd = 1:18044;
             net.divideParam.valInd = 18045:21117;
             net.divideParam.testInd = 21118:25517;
             net.trainParam.showWindow=0;
             [net,tr] = train(net,data_X,target_data);
             TestOut = net(data_X(:,21118:25517));
             %perf = perform(net,target_2_test,TestOut(1,:));
             [X,Y,T,AUC]=perfcurve(target_data(:,21118:25517),TestOut(1,:),1);
              
             %contour3(mx,my,perf);
             if (temp < AUC)
                 temp = AUC;
                 mx_f = mx;
                 my_f = my;
             end
    fprintf('finished')             
    end
end
 
net=patternnet([mx_f,my_f]);
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:18044;
net.divideParam.valInd = 18045:21117;
net.divideParam.testInd = 21118:25517;
[net,tr] = train(net,data_X,target_data);
TestOut = net(data_X(:,21118:25517));
%perf = perform(net,target_2_test,TestOut(1,:));
[X,Y,T,AUC]=perfcurve(target_data(:,21118:25517),TestOut(1,:),1);
figure; plot(X,Y);
figure; plotconfusion(target_data(:,21118:25517),TestOut);
threshold = 0.7;
Test2a = (TestOut(1,:) > threshold);
Test2b = (TestOut(2,:) > 1-threshold);
Test2=[Test2a;Test2b];
figure; plotconfusion(target_data(:,21118:25517),Test2);
