function generate(n_event_types, n_tokens_per_type, cluster_size, fname, n_seq, n_events)
% this function generates sequences for classification.  the sequences
% are distinguished by whether they contain a cluster of events with fixed lags.
%
% if fname is provided the results are stored in a file
% if n_examples is provided, the file will contain this many sequences
%
% n_event_types: # of different types of events
% n_tokens_per_type: # of different tokens of each type
% cluster_size: size of cluster (in # event types) that defines a positive
%               sequence
% fname: where to write output. if no file is specified, then the generated
%        sequence is graphed (Figure 1), and the sequences are merged and
%        fed into an inference procedure that tries to predict the event type
%        of the next event conditioned on the preceding events (Figure 2).
%        Use [] for output to screen
% n_seq: number of sequences to generate
% n_events: sequence length


if (nargin < 1)
   n_event_types = 5;
end
if (nargin < 2)
   n_tokens_per_type = 1;
end
if (nargin < 3)
   cluster_size = 2;
end
if (nargin < 4 || isempty(fname))
   fp = 2;
else
   fp = fopen(fname,'w');
end
if (nargin < 5)
   n_seq = 1;
end
if (nargin < 6)
   n_events = 100;
end

e = 1;
while (e <= n_seq)
   % generate n_events with poisson inter-arrival times
   s = unidrnd(n_event_types,1,n_events);
   dt = exprnd(1,[1 n_events]);
   dt(1) = 0;
   t = cumsum(dt);
   
   % look within all windows of cluster_size time steps and see if the
   % window includes event types 1:cluster_size
   
   event_separation = 10; % time units
   target_present_trial = (e/n_seq <= .5);
   ok_example = 1;
   if (target_present_trial)
      event_range = event_separation * (cluster_size-1);
      start_time = rand()*((max(t)-event_range)-(min(t)+event_range))+ ...
                                   (min(t)+event_range);
      s(n_events-cluster_size+1:n_events)=1:cluster_size;
      tgt_range=start_time + (0:event_separation:(cluster_size-1)*event_separation);
      t(n_events-cluster_size+1:n_events)=tgt_range;
      [~,ix] = sort(t);
      t = t(ix);
      s = s(ix);
   else
      if (closest_err(event_separation,s,t) < 1) % how close to a match?
         ok_example = 0; 
         fprintf('%d\n',e);
      end
   end
   
   if (ok_example)
      if (fp == 2)
         plot_stream(s,t);
         if (target_present_trial > 0 && fp==2)
            r=rectangle('position',[tgt_range(1) .8 tgt_range(end) .2]);
            r.FaceColor = 'k';
         end
      end

      m.in = s;
      m.tin = [0 t(2:end)-t(1:end-1)];
      m.out = [zeros(1,length(s)-1) target_present_trial*2-1];
      m.tout = [t(2:end)-t(1:end-1) 1];
      save_complete(fp,e,m)
      e = e + 1;
   end
end
   
%    if (fp ~= 2)
%       if (extrapolation_test)
%          save_extrapolation(fp,fp2,e,m);
%       else
%          save_complete(fp,e,m);
%       end
%    else  
%       halflives = halflives_save{e};
%       gammas = 1 ./ halflives;
%       alphas = alpha_const * gammas;
%       mus = -log(1-baseline_rate) * ones(size(halflives)); 
%       plot_streams(n_event_streams, out_save{e}, mus, gammas, alphas, all_halflives, halflives, m);
%    end
% end
% if (fp ~= 2)
%    fclose(fp);
%    if (extrapolation_test)
%       fclose(fp2);
%    end
% end

function plot_stream(s,t)
hold off
c = jet(max(s));
for i = 1:length(s)
   plot(t(i)*[1 1], [0 1],'color',c(s(i),:));
   hold on
end
set(gca,'xlim',[0 max(t)])
set(gca,'ylim',[0 1])

function save_complete(fp,e,m)
   % save in file or to screen
   fprintf(fp,'%03d', e);
   fprintf(fp,' %d', m.in);
   fprintf(fp,'\n%03d', e);
   fprintf(fp,' %.3f', m.tin);
   fprintf(fp,'\n%03d', e);
   fprintf(fp,' %d', m.out);
   fprintf(fp,'\n%03d', e);
   fprintf(fp,' %.3f', m.tout);
   fprintf(fp,'\n'); 

function save_extrapolation(fp,fp2,e,m) 
   l = floor(length(m.in)/2);
   % save training
   fprintf(fp,'%03d', e);
   fprintf(fp,' %d', m.in(1:l));
   fprintf(fp,'\n%03d', e);
   fprintf(fp,' %.3f', m.tin(1:l));
   fprintf(fp,'\n%03d', e);
   fprintf(fp,' %d', m.out(1:l));
   fprintf(fp,'\n%03d', e);
   fprintf(fp,' %.3f', m.tout(1:l));
   fprintf(fp,'\n'); 
   % save testing
   n = m;
   n.out(1:l)=0; % don't care value
   fprintf(fp2,'%03d', e);
   fprintf(fp2,' %d', n.in);
   fprintf(fp2,'\n%03d', e);
   fprintf(fp2,' %.3f', n.tin);
   fprintf(fp2,'\n%03d', e);
   fprintf(fp2,' %d', n.out);
   fprintf(fp2,'\n%03d', e);
   fprintf(fp2,' %.3f', n.tout);
   fprintf(fp2,'\n'); 

function best_err = closest_err(event_separation,in,t)

best_err = 1e10;
ix1 = find(in==1);
ix2 = find(in==2);
for j = 1:length(ix1)
   for k = 1:length(ix2)
      deltat = t(ix2(k))-t(ix1(j));
      if (abs(deltat-10) < best_err)
         best_deltat = deltat;
         best_err = abs(deltat-10);
      end
   end
end
