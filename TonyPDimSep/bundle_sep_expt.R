# normalize a vector
z <- function(x) x * 1/sqrt(sum(x^2))
# replace occurrences of v in x with w
repv <- function(x, v, w) replace(x, which(x==v), w)
# return a vector of mean and sd of x
meansd <- function(x, na.rm=TRUE) {
  c(mean=mean(x, na.rm=na.rm), sd=sd(x, na.rm=na.rm))
}
# return a list with mean, sd, var, and length of x
meansdl <- function(x, na.rm=TRUE) {
  list(u=mean(x, na.rm=na.rm), s=sd(x, na.rm=na.rm), v=sd(x, na.rm=na.rm)^2, n=sum(!is.na(x)))
}
# add names to x
named <- function(x, names_=NULL) {names(x) <- names_; x}

optimal_threshold <- function(signal_stats, expts) {
  decision_stats <- function(expts, threshold, details=FALSE) {
    decisions <- sapply(expts, function(expt)
      c(FN=sum(expt$pos < threshold), FP=sum(expt$neg >= threshold),
        n_pos=length(expt$pos), n_neg=length(expt$neg),
        n=length(expt$pos) + length(expt$neg)))
    p_wrong_decision <- (sum(decisions['FP',]) + sum(decisions['FN',])) / sum(decisions['n',])
    p_wrong_expt <- sum(colSums(decisions[c('FP', 'FN'),])>0) / ncol(decisions)
    if (details) {
      return(list(threshold=threshold,
                  p_wrong_decision=p_wrong_decision,
                  p_wrong_expt=p_wrong_expt,
                  n_expts=length(expts),
                  n_decisions=sum(decisions['n',]),
                  n_FP=sum(decisions['FP',]),
                  n_FN=sum(decisions['FN',]),
                  p_FP=sum(decisions['FP',]) / sum(decisions['n_neg',]),
                  p_FN=sum(decisions['FN',]) / sum(decisions['n_pos',])))
    } else {
      return(p_wrong_decision)
    }
  }
  opt <- optimize(function(tt) decision_stats(expts, threshold=tt), lower=0, upper=1)
  res <- rbind(opt=unlist(decision_stats(expts, threshold=opt$minimum, details=TRUE)),
               mid=unlist(decision_stats(expts, threshold=mean(c(signal_stats$neg$u, signal_stats$pos$u)), details=TRUE)))
  return(res)
}

run_expts <- function(mem, M, K, n_trials) {
  expts <- lapply(seq(n_trials), function(i) {
    j <- sample(M, K) # indices to include in the bundle
    x <- z(rowSums(mem[,j])) # the bundle, normalized
    y <- x %*% mem # dot product of x with every vector in mem
    return(list(pos=y[j], neg=y[-j]))
  })
  return(expts)
}

bundle_sep_expt <- function(
    n = 64,   # dimensionality of vectors
    M = 1000, # number of vectors in memory
    K = 4,    # number of vectors in a bundle
    n_trials = 1000,# number of trials
    layout = NULL,
    plot_ = NULL)
{
  # mem is a matrix of M normalized vectors of dim n, one vector per column
  mem <- apply(matrix(rnorm(M * n), ncol=M, dimnames=list(NULL, paste0('v', seq(M)))), 2, z)
  expts <- run_expts(mem, M, K, n_trials)
  ss <- list(neg=meansdl(c(sapply(expts, '[[', 'neg'))),
             pos=meansdl(c(sapply(expts, '[[', 'pos'))))
  # n_expt should equal n_trials
  param <- c(n=n, M=M, K=K, n_trials=n_trials, n_expt=length(expts))
  opt_decision <- optimal_threshold(signal_stats=ss, expts=expts)
  plot_bundle_sep_expt(expts, mem, sep=with(ss, (pos$u-neg$u)/sqrt(neg$v + pos$v)),
    main='vanilla bundle decoding experiment',
    param=param, opt_decision=opt_decision, layout=layout, plot_=plot_)
  return(list(param=param,
    res=with(ss, data.frame(row.names=c('neg','pos','sep'),
      u=c(neg$u, pos$u, pos$u-neg$u),
      sd=c(neg$s, pos$s, sqrt(neg$v+pos$v)),
      z=c(neg$u/neg$s, pos$u/pos$s, (pos$u-neg$u)/sqrt(neg$v + pos$v)))),
    opt_decision=opt_decision))
}

# version with linear dimensionality expansion
bundle_sep_lin_dimex_expt <- function(
    n = 64,    # dimensionality of vectors in higher-d space
    ns = 16,   # dimensionality of vectors in lower-d space
    M = 1000,  # number of vectors in memory
    K = 4,     # number of vectors in a bundle
    n_trials = 1000, # number of trials
    layout = NULL,
    plot_ = NULL)
{
  # X is a matrix to transform from ns to n: [K x ns] [ns x n] -> [K x n]
  X <- svd(matrix(rnorm(n * ns, sd=sqrt(1/n)), nrow=n))$u
  # ms is the collection of vectors in the low-d embedding space
  ms <- apply(matrix(rnorm(M * ns), ncol=M, dimnames=list(NULL, paste0('v', seq(M)))), 2, z)
  # mem is the collection of vectors in the high-d space
  mem <- apply(X %*% ms, 2, z)
  expts <- run_expts(mem, M, K, n_trials)
  ss <- list(neg=meansdl(c(sapply(expts, '[[', 'neg'))), pos=meansdl(c(sapply(expts, '[[', 'pos'))))
  param <- c(n=n, M=M, K=K, n_trials=n_trials, ns=ns, n_expt=length(expts))
  opt_decision <- optimal_threshold(signal_stats=ss, expts=expts)
  plot_bundle_sep_expt(expts, ms, mem, sep=with(ss, (pos$u-neg$u)/sqrt(neg$v + pos$v)),
                       main='bundle decoding on linear dimex',
                       param=param, opt_decision=opt_decision, layout=layout, plot_=plot_)
  return(list(param=param,
    res=with(ss, data.frame(row.names=c('neg','pos','sep'),
      u=c(neg$u, pos$u, pos$u-neg$u),
      sd=c(neg$s, pos$s, sqrt(neg$v+pos$v)),
      z=c(neg$u/neg$s, pos$u/pos$s, (pos$u-neg$u)/sqrt(neg$v + pos$v)))),
    opt_decision=opt_decision))
}

# version with non-linear dimensionality expansion
# dimensionality expansion is implemented via collapsing an outer product
bundle_sep_nonlin_dimex1_expt <- function(
    n = 64,    # dimensionality of vectors in higher-d space
    ns = 16,   # dimensionality of vectors in lower-d space
    order = 2, # order to use the dimensionality expansion
    M = 1000,  # number of vectors in memory
    K = 4,     # number of vectors in a bundle
    n_trials = 1000, # number of trials
    layout = NULL,
    plot_ = NULL)
{
  ms <- apply(matrix(rnorm(M * ns), ncol=M, dimnames=list(NULL, paste0('v', seq(M)))), 2, z)
  collapse_idx <- sample(rep(seq(len=n), length=ns^order))
  dimexp <- function(x) {
    y <- c(outer(x, x))
    for (j in seq(len=order-2))
      y <- c(outer(x, y))
    return(tapply(y, collapse_idx, sum))
  }
  mem <- apply(ms, 2, function(x) z(dimexp(x)))
  expts <- run_expts(mem, M, K, n_trials)
  ss <- list(neg=meansdl(c(sapply(expts, '[[', 'neg'))), pos=meansdl(c(sapply(expts, '[[', 'pos'))))
  param <- c(n=n, M=M, K=K, n_trials=n_trials, ns=ns, n_expt=length(expts), order=order)
  opt_decision <- optimal_threshold(signal_stats=ss, expts=expts)
  plot_bundle_sep_expt(expts, ms, mem, sep=with(ss, (pos$u-neg$u)/sqrt(neg$v + pos$v)),
                       main='bundle decoding on non-linear dimex v1',
                       param=param, opt_decision=opt_decision, layout=layout, plot_=plot_)
  return(list(param=param,
    res=with(ss, data.frame(row.names=c('neg','pos','sep'),
      u=c(neg$u, pos$u, pos$u-neg$u),
      sd=c(neg$s, pos$s, sqrt(neg$v+pos$v)),
      z=c(neg$u/neg$s, pos$u/pos$s, (pos$u-neg$u)/sqrt(neg$v + pos$v)))),
    opt_decision=opt_decision))
}

# version with V2 non-linear dimensionality expansion
# dimensionality expansion is implemented via collapsing random products of given order
bundle_sep_nonlin_dimex2_expt <- function(
    n = 64,    # dimensionality of vectors in higher-d space
    ns = 16,   # dimensionality of vectors in lower-d space
    order = 2, # order to use the dimensionality expansion
    J = 6,     # number of points in each dimensionality-explanding sum
    M = 1000,  # number of vectors in memory
    K = 4,     # number of vectors in a bundle
    n_trials = 1000, # number of trials
    layout = NULL,
    plot_ = NULL)
{
  ms <- apply(matrix(rnorm(M * ns), ncol=M, dimnames=list(NULL, paste0('v', seq(M)))), 2, z)
  collapse_idx <- sample(rep(seq(n), length=n*J))
  expand_idx <- lapply(seq(len=order), function(i) sample(rep(seq(ns), length=n*J)))
  dimexp <- function(x) {
    y <- x[expand_idx[[1]]]
    for (j in seq(2, order))
      y <- y * x[expand_idx[[j]]]
    return(tapply(y, collapse_idx, sum))
  }
  mem <- apply(ms, 2, function(x) z(dimexp(x)))
  expts <- run_expts(mem, M, K, n_trials)
  ss <- list(neg=meansdl(c(sapply(expts, '[[', 'neg'))), pos=meansdl(c(sapply(expts, '[[', 'pos'))))
  param <- c(n=n, M=M, K=K, n_trials=n_trials, ns=ns, n_expt=length(expts), order=order, J=J)
  opt_decision <- optimal_threshold(signal_stats=ss, expts=expts)
  plot_bundle_sep_expt(expts, ms, mem, sep=with(ss, (pos$u-neg$u)/sqrt(neg$v + pos$v)),
                       main='bundle decoding on non-linear dimex v2',
                       param=param, opt_decision=opt_decision, layout=layout, plot_=plot_)
  return(list(param=param,
    res=with(ss, data.frame(row.names=c('neg','pos','sep'),
      u=c(neg$u, pos$u, pos$u-neg$u),
      sd=c(neg$s, pos$s, sqrt(neg$v+pos$v)),
      z=c(neg$u/neg$s, pos$u/pos$s, (pos$u-neg$u)/sqrt(neg$v + pos$v)))),
    opt_decision=opt_decision))
}

# version with mix of lin & V2 non-linear dimensionality expansion
bundle_sep_nonlin_dimex3_expt <- function(
    n = 64,    # dimensionality of vectors in higher-d space
    ns = 16,   # dimensionality of vectors in lower-d space
    dxa = 0.5, # mixing weight of non-linear
    order = 2, # order to use the dimensionality expansion
    J = 6,     # number of points in each dimensionality-explanding sum
    M = 1000,  # number of vectors in memory
    K = 4,     # number of vectors in a bundle
    n_trials = 1000, # number of trials
    layout = NULL,
    plot_ = NULL)
{
  ms <- apply(matrix(rnorm(M * ns), ncol=M, dimnames=list(NULL, paste0('v', seq(M)))), 2, z)
  collapse_idx <- sample(rep(seq(n), length=n*J))
  expand_idx <- lapply(seq(len=order), function(i) sample(rep(seq(ns), length=n*J)))
  dimexp <- function(x) {
    y <- x[expand_idx[[1]]]
    for (j in seq(2, order))
      y <- y * x[expand_idx[[j]]]
    return(tapply(y, collapse_idx, sum))
  }
  mem_b <- apply(ms, 2, function(x) z(dimexp(x)))
  # v-matrix [K x ns] [ns x n] -> [K x n]
  X <- svd(matrix(rnorm(n * ns, sd=sqrt(1/n)), nrow=n))$u
  mem_a <- apply(X %*% ms, 2, z)
  # mix together
  mem <- apply((1-dxa) * mem_a + dxa * mem_b, 2, z)
  expts <- run_expts(mem, M, K, n_trials)
  ss <- list(neg=meansdl(c(sapply(expts, '[[', 'neg'))), pos=meansdl(c(sapply(expts, '[[', 'pos'))))
  param <- c(n=n, M=M, K=K, n_trials=n_trials, ns=ns, n_expt=length(expts), order=order, J=J, dxa=dxa)
  opt_decision <- optimal_threshold(signal_stats=ss, expts=expts)
  plot_bundle_sep_expt(expts, ms, mem, sep=with(ss, (pos$u-neg$u)/sqrt(neg$v + pos$v)),
    main='bundle decoding on non-linear dimex v3',
    param=param, opt_decision=opt_decision, layout=layout, plot_=plot_)
  return(list(param=param,
    res=with(ss, data.frame(row.names=c('neg','pos','sep'),
      u=c(neg$u, pos$u, pos$u-neg$u),
      sd=c(neg$s, pos$s, sqrt(neg$v+pos$v)),
      z=c(neg$u/neg$s, pos$u/pos$s, (pos$u-neg$u)/sqrt(neg$v + pos$v)))),
    opt_decision=opt_decision))
}

plot_bundle_sep_expt <- function(
  res, m_small, m_large=NULL,
  main='bundle decoding experiment', param=NULL,
  opt_decision=NULL, layout=NULL, sep=NULL, plot_=NULL)
{
  # layout(rbind(c(1,1,2), c(1,1,3)))
  if (is.null(layout))
    layout <- rbind(c(1,2), c(4,3))
  if (is.null(plot_))
    plot_ <- 1:4
  if (!is.character(layout) || layout!='n')
    graphics::layout(layout)
  par(cex=0.6)
  h_neg <- hist(sapply(res, '[[', 'neg'), breaks=seq(-1,1,len=201), plot=F)
  h_pos <- hist(sapply(res, '[[', 'pos'), breaks=seq(-1,1,len=201), plot=F)
  if (is.element(1, plot_)) {
    if (length(param)) {
      i <- seq(along=param)[!(names(param) %in% c('n_expt'))]
      while (length(i)) {
        main <- paste(main, '\n', paste(sapply(names(param)[head(i, 5)],
          function(n) sprintf('%s=%g', n, param[n])), collapse=' '))
        i <- tail(i, -5)
      }
    }
    with(h_neg,
         plot(x=mids, y=repv(density, 0, NA), type='h', col='red', lwd=1, main=main,
              ylim=range(c(h_neg$density, h_pos$density)),
              xlab=if (is.null(m_large)) 'cosines' else 'cosines (in hi-d space)',
              ylab='density'))
    with(h_pos,
      points(x=mids, y=repv(density, 0, NA), type='h', col='green', lwd=1))
    legend('topleft', legend=c('pos', 'neg'), pch=15, col=c('green', 'red'), bty='n')
    if (!is.null(sep))
      mtext(outer=F, line=-4, text=sprintf('  sep = %.2f Z', sep), cex=0.7, adj=0, padj=0)
    if (!is.null(opt_decision)) {
      abline(v=opt_decision[, 'threshold'], col='black', lwd=0.25)
      pct_correct <- sapply(100 * (1 - opt_decision[, c('p_wrong_expt')]), format, digits=3, scientific=F, nsmall=1)
      mtext(outer=F, line=0, cex=0.6, adj=0.5,
            text=sprintf('p(correct): mid:%s%% opt:%s%%', pct_correct['mid'], pct_correct['opt']))
    }
  }
  if (is.element(3, plot_)) {
    qq <- qqnorm(sapply(res, '[[', 'neg'), plot=FALSE)
    if (length(qq$x) > 2000) {
      i <- order(qq$y)
      j <- unique(round(approx(qq$x[i], seq(len=length(qq$x)), xout=seq(min(qq$x),max(qq$x),len=1000), rule=2)$y))
      qq$y <- qq$y[i][j]
      qq$x <- qq$x[i][j]
    }
    plot(qq$x, qq$y, col='red', xlab='Theoretical quantiles', ylab='Sample quantiles', main='Signal for neg membership\n(cosine)')
    qqline(sapply(res, '[[', 'neg'))
  }
  if (is.element(4, plot_)) {
    qq <- qqnorm(sapply(res, '[[', 'pos'), plot=FALSE)
    if (length(qq$x) > 2000) {
      i <- order(qq$y)
      j <- unique(round(approx(qq$x[i], seq(len=length(qq$x)), xout=seq(min(qq$x),max(qq$x),len=1000), rule=2)$y))
      qq$y <- qq$y[i][j]
      qq$x <- qq$x[i][j]
    }
    plot(qq$x, qq$y, col='green', xlab='Theoretical quantiles', ylab='Sample quantiles', main='Signal for pos membership\n(cosine)')
    qqline(sapply(res, '[[', 'pos'))
  }
  if (!is.null(m_large) && is.element(2, plot_)) {
    cp_small <- crossprod(m_small)
    cp_large <- crossprod(m_large)
    smoothScatter(cp_small, cp_large, xlab='cos in low-d space', ylab='cos in high-d space',
      main='Similarity mapping low-d -> high-d')
    cp_smooth <- loess.smooth(cp_small, cp_large, span=1/20, evaluation=200)
    with(cp_smooth, lines(x, y, col='yellow'))
    mtext(outer=F, line=-1, text=sprintf('Cor = %.2f %%', 100*cor(c(cp_small), c(cp_large))), cex=0.5, adj=0.02)
    p <- approx(x=cp_smooth$x, y=cp_smooth$y, xout=c(cp_small), rule=2)$y
    v1 <- mean((c(cp_large) - mean(c(cp_large)))^2)
    v2 <- mean((c(cp_large) - p)^2)
    mtext(outer=F, line=-2, text=sprintf('R^2(lo->hi) = %.2f %%', 100*(v1-v2)/v1), cex=0.5, adj=0.02)
    cp_smooth <- loess.smooth(cp_large, cp_small, span=1/20, evaluation=200)
    p <- approx(x=cp_smooth$x, y=cp_smooth$y, xout=c(cp_large), rule=2)$y
    v1 <- mean((c(cp_small) - mean(c(cp_small)))^2)
    v2 <- mean((c(cp_small) - p)^2)
    mtext(outer=F, line=-3, text=sprintf('R^2(hi->lo) = %.2f %%', 100*(v1-v2)/v1), cex=0.5, adj=0.02)
  }
  return(invisible(NULL))
}
