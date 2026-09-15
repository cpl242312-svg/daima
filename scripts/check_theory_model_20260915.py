"""Check mathematical identities only; no empirical data or estimated results."""

from math import exp, expm1, isclose, sqrt
from pathlib import Path
import re


def solution(q, rho=1.0, profit=1.0, lam=1.0, gamma=1.0):
    j = profit / rho
    theta = lam * lam * q * q / gamma
    x = 2 * rho * j / (rho + sqrt(rho * rho + 2 * rho * theta * j))
    effort = lam * q * x / gamma
    hazard = theta * x
    return j - x, x, effort, hazard


def derivative(fun, value, step=1e-5):
    return (fun(value + step) - fun(value - step)) / (2 * step)


def close(actual, expected):
    assert isclose(actual, expected, rel_tol=2e-6, abs_tol=2e-8), (
        actual, expected
    )


def check():
    count = 0
    for rho in (0.1, 1.0, 3.0):
        for profit in (0.2, 1.0, 4.0):
            for lam in (0.3, 1.0, 2.0):
                for gamma in (0.2, 1.0, 5.0):
                    for q in (0.1, 0.4, 0.8):
                        w, x, e, h = solution(q, rho, profit, lam, gamma)
                        j = profit / rho
                        assert 0 < x < j and 0 < w < j and e > 0 and h > 0
                        close(rho * w, -gamma * e * e / 2 + h * x)
                        close(gamma * e, lam * q * x)
                        close(h, sqrt(rho * rho + 2 * rho * lam**2 * q**2 * j / gamma) - rho)
                        targets = (
                            h * x / ((rho + h) * q),
                            -h * x / ((rho + h) * q),
                            e * rho / ((rho + h) * q),
                            h * (h + 2 * rho) / ((rho + h) * q),
                        )
                        for index, target in enumerate(targets):
                            close(derivative(lambda v: solution(v, rho, profit, lam, gamma)[index], q), target)
                        assert derivative(lambda v: -expm1(-2 * solution(v, rho, profit, lam, gamma)[3]), q) > 0
                        count += 1

    # Shared W in competing channels: higher relative sensitivity is not
    # a ranking of absolute probability increments.
    def channels(phi):
        qr = 0.7 + 0.05 * (phi - 0.5)
        qn = 0.07 + 0.014 * (phi - 0.5)
        theta_r, theta_n = qr * qr, qn * qn
        theta = theta_r + theta_n
        x = 2 / (1 + sqrt(1 + 2 * theta))
        h = theta * x
        total = -expm1(-h)
        return theta_r / theta * total, theta_n / theta * total

    pr, pn = channels(0.5)
    dr = derivative(lambda v: channels(v)[0], 0.5)
    dn = derivative(lambda v: channels(v)[1], 0.5)
    assert 0.014 / 0.07 > 0.05 / 0.7
    assert 0 < dn < dr
    close(dn / pn - dr / pr, 2 * (0.014 / 0.07 - 0.05 / 0.7))

    # Density saturation reverses the country interaction even when U_z > 0.
    tau, omega = 0.8, 0.8
    effect = lambda om: 4 * om * exp(-4 * tau * om)
    cross = derivative(effect, omega)
    close(cross, 4 * exp(-4 * tau * omega) * (1 - 4 * tau * omega))
    assert cross < 0

    # Both negative and positive total resistance effects are possible.
    phi, value, phi_prime = 0.6, 2.0, 0.2
    assert value * phi_prime + phi * (-0.2) > 0
    assert value * phi_prime + phi * (-1.0) < 0

    path = Path(__file__).resolve().parents[1] / 'docs' / '理论模型_顶刊修改版.md'
    text = path.read_text()
    tags = [int(n) for n in re.findall(r'\\tag\{(\d+)\}', text)]
    assert tags == list(range(1, 24)), tags
    assert text.count('$$') == 46
    for name in ('Barrot', 'Bernard', 'Carvalho', 'Elliott', 'Farrell', 'Ferrari',
                 'Grossman', 'Katz', 'Macchiavello', 'Nunn', 'Schmidt'):
        body, refs = text.split('## 参考文献')
        assert name in body and name in refs, name
    print(f'PASS: {count} positive-parameter Bellman checks; finite-difference identities; 23 equation tags; 11 reference cross-checks.')
    print(f'Counterexample: new-client probability derivative={dn:.8f}; original-client derivative={dr:.8f}; relative new-client sensitivity is higher.')
    print(f'Counterexample: destination-fit interaction on resistance effect={cross:.8f}, despite increasing threshold U(z)=z.')
    print('These are mathematical validation inputs, not calibrated parameters or empirical findings.')


if __name__ == '__main__':
    check()
