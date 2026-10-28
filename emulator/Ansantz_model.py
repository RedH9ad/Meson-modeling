from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch

DeviceLike = Union[str, torch.device, None]

DEFAULT_DTYPE = torch.float64

def get_device(device: DeviceLike = None) -> torch.device:
    """
    Определяет torch.device для вычислений.
      - device=None       -> автоматически: cuda, если доступна, иначе cpu
      - device="cuda"/... -> использовать явно указанное устройство
    """
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

@dataclass
class ModelParams:
    """
    Вектор параметров V = {m, n, p, a, b, d, c, k} + гравитационная
    константа G5 (нужна только для энтропии s_BH, формула 1.17).
    Хранится как обычные python-float (device-agnostic); тензоры на нужном
    устройстве создаются уже внутри HolographicBackground.

    A(z) = m*log(1+a z^2) + n*log(1+b z^4) + p*log(1+d z^6)     (1.7)
    f(z) = exp(c z^2 - A(z) + k)                                 (1.8)

    Поддержка ПРОИЗВОЛЬНОГО анзаца A(z)
    ------------------------------------
    Поле `A_fn`, если задано, ПОЛНОСТЬЮ ЗАМЕНЯЕТ формулу (1.7): вместо неё
    HolographicBackground.A(z) вызывает A_fn(z) (произвольная
    autograd-дифференцируемая функция torch.Tensor -> torch.Tensor). Поля
    m,n,p,a,b,d в этом случае игнорируются (можно оставить нулями) - остаются
    актуальными только c, k, G5. Это и есть механизм "разных анзацев A(z)",
    запрошенный в задании: любую функцию A(z) (не обязательно из семейства
    трёх логарифмов) можно исследовать, не трогая остальной код класса -
    производные A'(z), A''(z), профиль дилатона phi(z), g(z), T(zh,mu),
    s_BH, F[zh,mu] будут автоматически пересчитаны через тот же самый
    torch.autograd конвейер. См. также from_custom_A() и from_dilaton()
    ниже - два готовых способа заполнить это поле.

    """
    m: float
    n: float
    p: float
    a: float
    b: float
    d: float
    c: float
    k: float
    G5: float = 1.0
    A_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None


def from_chen_huang(d: float, a: float, b: float, c: float, k: float,
                     G5: float) -> ModelParams:
    """
    Переводит параметризацию Chen & Huang (arXiv:2405.06179, формула 2.24):
        A(z) = d*log(1+a z^2) + d*log(1+b z^4)
    в общий анзац (1.7), где это соответствует m = n = d, p = 0.

    ВАЖНО - ЗНАК c (исправленная несогласованность):
    Статья Chen & Huang пишет свои формулы g(z), T(zh,mu) (её ур. 2.19,
    2.21, 2.22) буквально через exp(-c*z^2) с её же табличными (обычно
    ОТРИЦАТЕЛЬНЫМИ) значениями c из Table 1. Однако внутренний движок этого
    файла (HolographicBackground.g/_T_raw/_J_integrand) использует
    ПРОТИВОПОЛОЖНОЕ соглашение exp(+c*z^2) (см. docstring модуля - выбор
    ветви установлен прямым численным сканированием экстремумов T(zh) и
    самосогласованно проверен только по бенчмаркам CEP самой статьи
    Chen & Huang). Чтобы параметризация Chen-Huang осталась физически
    согласованной внутри движка exp(+c*z^2), табличное c статьи нужно
    ПОДСТАВЛЯТЬ СО ЗНАКОМ МИНУС:
        c_internal = -c_paper.
    Численно проверено (бисекция критической точки по числу экстремумов
    T(zh)): с этой заменой знака воспроизводятся ВСЕ ТРИ табличных CEP
    статьи (Table 1, раздел 6) с точностью <1%:
        Nf=2:     mu_c = 0.461 GeV (статья: 0.46),  T_c = 0.147 GeV (статья: 0.147)
        Nf=2+1:   mu_c = 0.745 GeV (статья: 0.74),  T_c = 0.094 GeV (статья: 0.094)
        Nf=2+1+1: mu_c = 0.877 GeV (статья: 0.87),  T_c = 0.107 GeV (статья: 0.108)
    А для Nf=0 (c=0, знак несущественен - химпотенциал вообще не входит в
    формулы) воспроизводится T_c = 0.266 GeV методом Максвелла (статья: 0.265).
    Без этой замены знака (буквальная подстановка табличного c статьи)
    кривые T(zh) для Nf=2,2+1,2+1+1 остаются монотонными при mu=mu_c и
    фазовый переход первого рода не появляется вовсе - т.е. это физически
    значимая, а не "стилистическая" поправка.
    """
    return ModelParams(m=d, n=d, p=0.0, a=a, b=b, d=0.0, c=-c, k=k, G5=G5)


# Таблица параметров из статьи (arXiv:2405.06179, Table 1) для разных числа
# ароматов Nf - знак c уже исправлен внутри from_chen_huang() (см. её docstring).
NF0_PARAMS = from_chen_huang(d=-0.584, a=0.0, b=0.072, c=0.0, k=0.0, G5=1.326)
NF2_PARAMS = from_chen_huang(d=-0.382, a=0.067, b=0.023, c=-0.377, k=0.0, G5=0.885)
NF21_PARAMS = from_chen_huang(d=-0.173, a=0.204, b=0.013, c=-0.264, k=-0.824, G5=0.400)
NF211_PARAMS = from_chen_huang(d=-0.171, a=0.196, b=0.014, c=-0.362, k=-0.735, G5=0.391)

# Реестр всех "именованных" моделей - удобно перебирать в цикле для
# сравнительных графиков/исследований разных Nf (см. демо-блок и
# build_notebook.py).
PRESETS = {
    "Nf=0 (чистая глюодинамика)": NF0_PARAMS,
    "Nf=2": NF2_PARAMS,
    "Nf=2+1": NF21_PARAMS,
    "Nf=2+1+1": NF211_PARAMS,
}


def from_custom_A(A_fn: Callable[[torch.Tensor], torch.Tensor],
                   c: float, k: float, G5: float) -> ModelParams:
    """
    Оборачивает ПРОИЗВОЛЬНУЮ функцию A(z) (torch.Tensor -> torch.Tensor,
    дифференцируемую через autograd) в ModelParams - самый прямой способ
    исследовать "другой анзац A(z)", не входящий в семейство трёх
    логарифмов (1.7). Поля m,n,p,a,b,d формально нулевые и не используются
    (см. ModelParams.A_fn).

    Пример
    ------
    >>> A_custom = lambda z: -2.0 * torch.log1p(0.05 * z**2 + 0.001 * z**4)
    >>> params = from_custom_A(A_custom, c=0.2, k=0.0, G5=1.0)
    >>> model = HolographicBackground(params)
    """
    return ModelParams(m=0.0, n=0.0, p=0.0, a=0.0, b=0.0, d=0.0,
                        c=c, k=k, G5=G5, A_fn=A_fn)

class ParametricAnsatzA(torch.nn.Module):
    """
    Обобщение формулы (1.7) с ОБУЧАЕМЫМИ параметрами, хранящимися в
    ОДНОМ векторе theta:

        A(z; theta) = sum_{i=1}^{N} coef_i * log(1 + scale_i * z^{e_i}),
        e = EXPONENTS = (2, 4, 6),  N = 3.

    Раскладка вектора (длина 2N = 6):

        theta = [coef_1, coef_2, coef_3, scale_1, scale_2, scale_3]
                 |------- coef -------|  |------- scale -------|

    При e=(2,4,6) это в точности семейство (1.7): coef=[m,n,p],
    scale=[a,b,d].

    Почему один вектор
    -------------------
    * ansatz.parameters() возвращает РОВНО ОДИН Parameter - theta, поэтому
      любой оптимизатор (torch.optim.Adam, scipy L-BFGS-B, Optuna-обёртки)
      работает с одним плоским вектором без ручной склейки групп;
    * coef и scale - это НЕ отдельные Parameter, а представления (views)
      theta[:N] и theta[N:]: градиент по ним автоматически попадает в
      theta.grad, а обновления theta (opt.step(), copy_) сразу видны в
      coef/scale;
    * EXPONENTS - фиксированная «форма» анзаца: хранится как buffer
      (переносится .to(device) вместе с модулем, но не оптимизируется).

    Физическое ограничение scale_i > 0
    -----------------------------------
    Аргумент 1 + scale_i * z^{e_i} обязан оставаться положительным, иначе
    A(z) перестаёт быть вещественной (см. check_physical_domain). В
    предыдущей версии это гарантировалось через softplus(raw_scale); теперь
    theta хранит scale НАПРЯМУЮ (физические значения), а ограничение
    обеспечивается:
      * в L-BFGS-B - явными bounds (scale_i in [1e-4, 1.0]);
      * в Adam - проекцией project_() после каждого шага (projected
        gradient descent), см. adam_calibrate_ansatz.

    Регулярность в нуле (A(0)=0, A'(0)=0) выполняется автоматически при
    любых coef, scale, т.к. все показатели e_i >= 2.

    Пример
    ------
    >>> theta0 = torch.tensor([-0.40, -0.35, 0.01, 0.08, 0.02, 0.001],
    ...                       dtype=DEFAULT_DTYPE, device="cuda")
    >>> ansatz = ParametricAnsatzA(theta0)
    >>> params = from_parametric_A(ansatz, c=0.42, k=-0.2, G5=0.85)
    >>> model = HolographicBackground(params, device="cuda")
    >>> optimizer = torch.optim.Adam(ansatz.parameters(), lr=1e-3)  # [theta]
    """

    EXPONENTS = (2.0, 4.0, 6.0)

    # минимальное допустимое значение scale_i (для проекции в Adam)
    SCALE_MIN = 1e-6

    def __init__(self, theta: torch.Tensor):
        super().__init__()
        self.theta_temp = theta

        if not isinstance(theta, torch.Tensor):
            raise TypeError(
                "theta должен быть torch.Tensor"
            )

        if theta.ndim != 1:
            raise ValueError(
                "theta должен быть одномерным"
            )

        self.n_terms = len(self.EXPONENTS)

        expected_size = 2 * self.n_terms

        # if theta.numel() != expected_size:
        #     raise ValueError(
        #         f"Ожидалось {expected_size} параметров, "
        #         f"получено {theta.numel()}."
        #     )

        if not torch.is_floating_point(theta):
            raise TypeError(
                "theta должен иметь floating-point dtype"
            )

        # Фиксированные степени — НЕ параметры оптимизации
        self.register_buffer(
            "exponents_buf",
            torch.tensor(
                self.EXPONENTS,
                dtype=theta.dtype,
                device=theta.device
            )
        )

        # Единственный обучаемый Parameter
        self.theta = torch.nn.Parameter(
            theta.detach().clone()
        )
        

    # ------------------------------------------------------------
    # Удобный конструктор из «человеческих» coef / scale
    # ------------------------------------------------------------

    @classmethod
    def from_coef_scale(
        cls,
        coef,
        scale,
        dtype: torch.dtype = DEFAULT_DTYPE,
        device: DeviceLike = None
    ) -> "ParametricAnsatzA":
        """theta = concat(coef, scale)."""
        device = get_device(device)
        theta = torch.tensor(
            [float(v) for v in coef] + [float(v) for v in scale],
            dtype=dtype,
            device=device
        )
        return cls(theta)

    @property
    def coef(self) -> torch.Tensor:
        """
        Представление первых N элементов theta.
        Не отдельный Parameter.
        """
        return self.theta[:self.n_terms]

    @property
    def scale(self) -> torch.Tensor:
        """
        Представление последних N элементов theta.
        Не отдельный Parameter.
        """
        return self.theta[self.n_terms:]

    @torch.no_grad()
    def project_(self) -> None:
        """
        Проекция на физическую область: scale_i >= SCALE_MIN.
        Вызывается после каждого шага градиентного оптимизатора без
        встроенных ограничений (Adam) - in-place, без участия autograd.
        """
        self.theta[self.n_terms:].clamp_(min=self.SCALE_MIN)

    def forward(
        self,
        z: torch.Tensor
    ) -> torch.Tensor:

        if not isinstance(z, torch.Tensor):
            z = torch.tensor(z, dtype=torch.double)
        

        z = z.to(
            dtype=self.theta.dtype,
            device=self.theta.device
        )

        z_pow = (
            z.unsqueeze(-1)
            ** self.exponents_buf
        )

        terms = (
            self.coef
            * torch.log1p(
                self.scale * z_pow
            )
        )

        return terms.sum(dim=-1)


def from_parametric_A(ansatz: "ParametricAnsatzA", c, k: float, G5: float,
                       device: DeviceLike = None) -> ModelParams:
    """
    Оборачивает ОБУЧАЕМЫЙ ParametricAnsatzA (единственный Parameter -
    вектор theta) в ModelParams - аналог from_custom_A(), но специально для
    случая, когда A(z) должна оставаться дифференцируемой ПО theta (не
    только по z).

    Если передан device, ansatz дополнительно переносится на него
    (ansatz.to(device)) - параметры анзаца и HolographicBackground(params,
    device=device) будут согласованы на одном устройстве.

    c, k могут быть python float или torch.Tensor с requires_grad=True -
    в последнем случае градиент по ним тоже течёт через весь пайплайн
    (HolographicBackground._t() не отрывает тензор от графа).
    """
    if device is not None:
        ansatz = ansatz.to(get_device(device))
    return ModelParams(m=0.0, n=0.0, p=0.0, a=0.0, b=0.0, d=0.0,
                        c=c, k=k, G5=G5, A_fn=ansatz)


class ChenHuangAnsatz(torch.nn.Module):
    """
    Chen--Huang ansatz (формула 2.24, arXiv:2405.06179):

        A(z; theta)
        =
        d * log(1 + a*z^2)
        + d * log(1 + b*z^4)

    Обучаемые параметры хранятся в одном тензоре:

        theta = [a, b, c, d, k]

    Соответствие индексов:

        theta[0] = a
        theta[1] = b
        theta[2] = c
        theta[3] = d
        theta[4] = k

    a, b, c, d, k - представления (views) элементов theta, а не отдельные
    Parameter: градиент по любому из них попадает в theta.grad, обновления
    theta сразу видны через свойства. c и k передаются в ModelParams как
    те же самые views (см. from_chen_huang_trainable), поэтому движок
    HolographicBackground видит актуальные значения после каждого шага
    оптимизатора.

    G5 НЕ обучается (не входит в T(zh,mu)) и хранится отдельно как buffer.

    Физическое ограничение a, b > 0 (аргументы логарифмов 1 + a z^2,
    1 + b z^4 положительны): в L-BFGS-B - явные bounds, в Adam - проекция
    project_() после каждого шага.

    Внутреннее соглашение для c:

        c_internal = -c_paper

    если параметры берутся из таблицы статьи (там exp(-c z^2), а движок
    использует exp(+c z^2)) - это делает from_paper_table() автоматически.

    Пример
    ------
    >>> ansatz = ChenHuangAnsatz.from_paper_table(
    ...     d=-0.382, a=0.067, b=0.023, c_paper=-0.377, k=0.0, G5=0.885)
    >>> params = from_chen_huang_trainable(ansatz, device="cuda")
    >>> model = HolographicBackground(params, device="cuda")
    >>> optimizer = torch.optim.Adam(ansatz.parameters(), lr=1e-3)  # [theta]
    """

    PARAMETER_NAMES = ("a", "b", "c", "d", "k")
    N_PARAMETERS = 5
    DEFAULT_G5: float = 0.885

    # индексы параметров, которые обязаны быть > 0 (a, b)
    POSITIVE_INDICES = (0, 1)
    POSITIVE_MIN = 1e-6

    def __init__(
        self,
        theta: torch.Tensor,
        G5: float = DEFAULT_G5
    ):
        super().__init__()
        self.theta_temp = theta

        # ====================================================
        # Проверка theta
        # ====================================================

        if not isinstance(theta, torch.Tensor):
            raise TypeError(
                "theta должен быть torch.Tensor"
            )

        if theta.ndim != 1:
            raise ValueError(
                "theta должен быть одномерным"
            )

        # if theta.numel() != self.N_PARAMETERS:
        #     raise ValueError(
        #         f"Ожидается {self.N_PARAMETERS} параметров "
        #         f"{self.PARAMETER_NAMES}, "
        #         f"получено {theta.numel()}."
        #     )

        if not torch.is_floating_point(theta):
            raise TypeError(
                "theta должен иметь floating-point dtype"
            )

        # ====================================================
        # Единственный обучаемый параметр
        # ====================================================

        self.theta = torch.nn.Parameter(
            theta.detach().clone()
        )

        # ====================================================
        # G5 — фиксированный параметр
        # ====================================================

        self.register_buffer(
            "G5",
            torch.tensor(
                float(G5),
                dtype=theta.dtype,
                device=theta.device
            )
        )

    # ========================================================
    # Представления элементов theta
    # ========================================================

    @property
    def a(self) -> torch.Tensor:
        return self.theta[0]

    @property
    def b(self) -> torch.Tensor:
        return self.theta[1]

    @property
    def c(self) -> torch.Tensor:
        return self.theta[2]

    @property
    def d(self) -> torch.Tensor:
        return self.theta[3]

    @property
    def k(self) -> torch.Tensor:
        return self.theta[4]

    # ========================================================
    # Инициализация из параметров статьи
    # ========================================================

    @classmethod
    def from_paper_table(
        cls,
        d: float,
        a: float,
        b: float,
        c_paper: float,
        k: float,
        G5: float,
        dtype: torch.dtype = DEFAULT_DTYPE,
        device: DeviceLike = None
    ) -> "ChenHuangAnsatz":

        device = get_device(device)

        theta = torch.tensor(
            [
                float(a),
                float(b),
                -float(c_paper),
                float(d),
                float(k),
            ],
            dtype=dtype,
            device=device
        )

        return cls(
            theta=theta,
            G5=G5
        )

    # ========================================================
    # Проекция на физическую область (для Adam)
    # ========================================================

    @torch.no_grad()
    def project_(self) -> None:
        """a, b >= POSITIVE_MIN (in-place, без участия autograd)."""
        for i in self.POSITIVE_INDICES:
            self.theta[i].clamp_(min=self.POSITIVE_MIN)

    # ========================================================
    # A(z)
    # ========================================================

    def A(
        self,
        z: torch.Tensor
    ) -> torch.Tensor:

        z = z.to(
            dtype=self.theta.dtype,
            device=self.theta.device
        )

        return (
            self.d * torch.log1p(
                self.a * z**2
            )
            +
            self.d * torch.log1p(
                self.b * z**4
            )
        )

    def forward(
        self,
        z: torch.Tensor
    ) -> torch.Tensor:

        return self.A(z)


def from_chen_huang_trainable(
    ansatz: "ChenHuangAnsatz",
    device: DeviceLike = None
) -> ModelParams:

    if device is not None:
        ansatz = ansatz.to(
            get_device(device)
        )

    return ModelParams(
        m=0.0,
        n=0.0,
        p=0.0,
        a=0.0,
        b=0.0,
        d=0.0,

        c=ansatz.c,
        k=ansatz.k,

        # G5 фиксирован, requires_grad=False
        G5=ansatz.G5,

        A_fn=ansatz
    )
