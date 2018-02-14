% i am writing this script because i forgot to reject 'negative' sequences
% in which the target separation occurred by accident.

x=load('disperse2.test');
x= x(:,2:end); % get rid of example index

bad_neg_example = 0;
good_neg_example = 0;
for i = 20001:4:size(x,1)
   in = x(i,:);
   t = cumsum(x(i+1,:));
   ix1 = find(in==1);
   ix2 = find(in==2);
   best_err = 1e20;
   for j = 1:length(ix1)
      for k = 1:length(ix2)
         deltat = t(ix2(k))-t(ix1(j));
         if (abs(deltat-10) < best_err)
            best_deltat = deltat;
            best_err = abs(deltat-10);
         end
      end
   end
   if (best_err<.5)
      bad_neg_example = bad_neg_example + 1;
   else
      good_neg_example = good_neg_example + 1;
   end
end
fprintf('good %d bad %d fraction bad %f\n',good_neg_example,bad_neg_example, ...
        bad_neg_example/(bad_neg_example+good_neg_example));
