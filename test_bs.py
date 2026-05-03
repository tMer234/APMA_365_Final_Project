#!/usr/bin/env python3

from black_scholes import black_scholes_call, black_scholes_put, black_scholes_implied_volatility

def main():
    # Test parameters
    S = 100    # Current stock price
    K = 105    # Strike price  
    T = 0.25   # Time to expiration (3 months)
    r = 0.05   # Risk-free rate (5%)
    sigma = 0.2 # Volatility (20%)
    
    # Calculate option prices
    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price = black_scholes_put(S, K, T, r, sigma)
    
    print("Black-Scholes Option Pricing Test")
    print("=" * 35)
    print(f"Stock Price (S):     ${S}")
    print(f"Strike Price (K):    ${K}")
    print(f"Time to Expiry (T):  {T} years")
    print(f"Risk-free Rate (r):  {r*100}%")
    print(f"Volatility (σ):      {sigma*100}%")
    print("-" * 35)
    print(f"Call Option Price:   ${call_price:.2f}")
    print(f"Put Option Price:    ${put_price:.2f}")

    implied_volatility = black_scholes_implied_volatility(call_price, S, K, T, r, 'call')
    print(f"Implied Volatility:   {implied_volatility*100}%")


if __name__ == "__main__":
    main()