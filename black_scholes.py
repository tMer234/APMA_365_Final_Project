import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def black_scholes_vega(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def black_scholes_implied_volatility(Market_Price, S, K, T, r, option_type='call', tol=1e-6, max_iter=100):
  
    # Set bounds based on option type
    if option_type.lower() == 'call':
        lower_bound = max(0, S - K * np.exp(-r * T))
        upper_bound = S
        price_fn = black_scholes_call
    elif option_type.lower() == 'put':
        lower_bound = max(0, K * np.exp(-r * T) - S)
        upper_bound = K * np.exp(-r * T)
        price_fn = black_scholes_put
    else:
        raise ValueError("option_type must be 'call' or 'put'")
   
    if not (lower_bound <= Market_Price <= upper_bound):
        raise ValueError("Market price is not in the range of the Black-Scholes model")
    #Manaster and Koehler (1982)
    initial_guess = np.sqrt(np.abs(np.log(Market_Price/S) + r * T) *(2/T))
    sigma = max(initial_guess, 1e-6)
    for i in range(max_iter):
       vega = black_scholes_vega(S, K, T, r, sigma)
       if np.isnan(vega) or np.isclose(vega, 0.0):
           raise ValueError("Failed to converge: near-zero vega")
       sigma = sigma - (price_fn(S, K, T, r, sigma) - Market_Price) / vega
       sigma = max(sigma, 1e-8)
       if abs(price_fn(S, K, T, r, sigma) - Market_Price) < tol:
           return sigma
    raise ValueError("Failed to converge")