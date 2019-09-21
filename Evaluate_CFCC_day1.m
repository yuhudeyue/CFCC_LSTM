function Evaluate_CFCC_day1(location1,location2,location3,location4)
%location1//end line
%location2//end column
%location3//start line
%location4//start column
% date_1=472;
% date_2=472;
% date_3=252;
% date_4=252;
ending = 451;
data_num = ending-3;
day1=ending+1;%start to predict
day2=ending+7;
day = day1;
load('testdata2007.mat');

Testdata(:,:,1:100)=testdata;
load('testdata2008.mat');

Testdata(:,:,101:200)=testdata;
load('testdata2009.mat');

Testdata(:,:,201:300)=testdata;
load('testdata2010.mat');

Testdata(:,:,301:400)=testdata;
load('testdata2011.mat');

Testdata(:,:,401:500)=testdata;
load('testdata2012.mat');

Testdata(:,:,501:600)=testdata;
testdata = Testdata;

%out_f=Evaluate1(location1,location2,location3,location4,testdata);
for i=location3:location1
    for j=location4:location2
          l=length(find(testdata(i-2:i+2,j-2:j+2,:)==0));
        if l==0
          out = test_lstm_day1(testdata(location1-2:location1+2,location2-2:location2+2,:),i,j,day-3,data_num);
        end
    end
end
predict2=Evaluate2_day1(location1,location2,location3,location4,testdata,day-1);
[A,mse] = Evaluate3(location1,location2,location3,location4,testdata,predict2,out,day1,day2);

end