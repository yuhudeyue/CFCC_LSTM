function Training_CFCC_day1(day1,day2,location1,location2,location3,location4)
%location1//end line
%location2//end column
%location3//start line
%location4//start column
% date_1=472;
% date_2=472;
% date_3=252;
% date_4=252;
%day1 = 1;
%day2 = 501;
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
day = day2-day1;
for i=location3
    for j=location4
          l=length(find(testdata(i-2:i+2,j-2:j+2,:)==0));
        if l==0
            Training_1_day1( location1,location2,location3,location4,testdata(location3-2:location3+2,location4-2:location4+2,1:day2),day-2);
        end
    end
end
weight_global = Training_2_day1( location1,location2,location3,location4,testdata(:,:,1:day2),day);

end