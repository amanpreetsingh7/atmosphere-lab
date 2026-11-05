"""Conditional likelihoods for detections and explicitly censored measurements."""
import numpy as np
from scipy.special import log_ndtr


def likelihood_grid(predictions,observations,covariance=None,model_fractional_error=0.):
    """Return relative likelihood, never an implicit posterior over irregular grid nodes."""
    p=np.asarray(predictions,float)
    if p.ndim!=2 or p.shape[1]!=len(observations) or not len(observations) or not len(p) or np.any(~np.isfinite(p)) or np.any(p<0):
        raise ValueError('Predictions must be a finite, nonnegative model-by-band matrix.')
    if not np.isfinite(model_fractional_error) or model_fractional_error<0:raise ValueError('Model error must be nonnegative.')
    values=np.asarray([r['contrast'] for r in observations],float)
    sigmas=np.asarray([r['sigma'] for r in observations],float)
    kinds=[r['kind'] for r in observations]
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(sigmas)) or np.any(sigmas<=0) or any(k not in ('detection','upper_limit') for k in kinds):
        raise ValueError('Each detection/censored upper limit needs a finite contrast and positive one-sigma measurement noise.')
    if any(k=='upper_limit' and v<=0 for k,v in zip(kinds,values)):raise ValueError('Upper-limit thresholds must be positive.')
    detections=np.array([k=='detection' for k in kinds]);logl=np.zeros(len(p))
    if covariance is not None:
        if not np.all(detections):raise ValueError('Correlated censoring is not supported. Covariance input requires detections in every band.')
        c=np.asarray(covariance,float)
        if c.shape!=(len(values),len(values)) or np.any(~np.isfinite(c)) or not np.allclose(c,c.T,rtol=1e-10,atol=0):raise ValueError('Covariance must be a finite symmetric band-by-band matrix in contrast squared.')
        if not np.allclose(np.diag(c),sigmas**2,rtol=1e-6,atol=0):raise ValueError('Covariance diagonal must equal the supplied sigma squared.')
        np.linalg.cholesky(c)
        for i,model in enumerate(p):
            total=c+np.diag((model_fractional_error*model)**2)
            chol=np.linalg.cholesky(total);resid=np.linalg.solve(chol,values-model)
            logl[i]=-.5*resid@resid-np.log(np.diag(chol)).sum()
    else:
        noise=np.sqrt(sigmas[None,:]**2+(model_fractional_error*p)**2)
        if np.any(detections):
            resid=(p[:,detections]-values[detections])/noise[:,detections]
            logl+=np.sum(-.5*resid**2-np.log(noise[:,detections]),axis=1)
        if np.any(~detections):
            logl+=log_ndtr((values[~detections]-p[:,~detections])/noise[:,~detections]).sum(axis=1)
    if not np.any(np.isfinite(logl)):raise ValueError('Likelihood is not finite for any model.')
    delta=logl-np.max(logl)
    return {'log_likelihood':logl.tolist(),'relative_likelihood':np.exp(delta).tolist(),
        'best_model_index':int(np.argmax(logl)),
        'method':'Gaussian detections; independent censored Gaussian upper limits P(noisy measurement < threshold | model). Optional detection covariance and user-declared fractional model error.',
        'interpretation':'Relative likelihood at tested grid points, conditional on the selected star, distance, model family, composition, response and evolutionary prescription. No prior or posterior normalization is imposed. Grid density is not a prior; this is not completeness or an occurrence constraint.'}
