"""The fixture-family generation pipeline: DSL schema, gates, frozen registry.

experiments/fixture_gen.py is the development-time trust mechanism behind
experiments/probes.py's fixture families: an LLM flow proposes a family as a
constrained DSL record and four deterministic gates measure it.  These tests
pin (1) the DSL schema (bad injections rejected before any gate runs),
(2) the frozen registry reproduces the pre-pipeline fixtures BITWISE — the
migration of the four hand-designed families is provenance, not new numbers,
(3) each gate does its own job (a nondeterministic witness, a
mean-shifting injection and a periodicity design that the band-24 probe
completes are all rejected, with numbers), and (4) a good proposal is
appended to the registry through the CLI's admit().

The zero-noise side of the trust rule (a lag-13 noise probe completing no
family, generated families included) lives in test_experiments_probes.py's
test_negative_controls_kill_noise_probes, which validates over the whole
registry.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REGISTRY_PATH = _REPO_ROOT / "experiments" / "specs" / "probe-fixture-registry.json"
_PROPOSALS_DIR = _REPO_ROOT / "experiments" / "specs" / "proposals"


def _load_fixture_gen() -> Any:
    """Load experiments/fixture_gen.py by file path (it loads probes itself)."""
    source = _REPO_ROOT / "experiments" / "fixture_gen.py"
    spec = importlib.util.spec_from_file_location("rhg_fixture_gen", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["rhg_fixture_gen"] = module
    spec.loader.exec_module(module)
    return module


G = _load_fixture_gen()
P = G.load_probes_module()
REGISTRY = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))


def family_entry(family_id: str) -> dict[str, Any]:
    entry = next((e for e in REGISTRY["families"] if e["familyId"] == family_id), None)
    assert entry is not None, f"no {family_id!r} family in the registry"
    return entry


def fixture_digest(artifacts: dict[str, Any]) -> str:
    """Stable digest of one fixture: e (NaN-padded), r, origins, activations, meta."""

    def h(array: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]

    residuals = artifacts["residuals"]
    parts = [h(residuals["e"]), h(residuals["r"]), h(residuals["origins"])]
    for name in sorted(artifacts["activations"]):
        parts.append(h(artifacts["activations"][name]))
    parts.append(hashlib.sha256(json.dumps(artifacts["meta"], sort_keys=True).encode()).hexdigest()[:16])
    return "+".join(parts)


# Digests of the fixtures the PRE-registry implementation produced (residual:
# seeds 0-2; activation: seed 0), captured before the rewiring and
# independently re-verified against git-HEAD probes.py.  The residual lists
# carry 10 entries in registry order: the first SIX are the historical seed
# families (the migration must reproduce them bitwise — provenance, not new
# numbers); the last four are the two pipeline-admitted families at their
# appended seed slots.
_EXPECTED_DIGESTS = json.loads(r"""{
 "residual/0/pairs": [
  "887ad8306eb5916f+ed0b443a6c204355+7d6c05abd13d3b81+d842fd5c8b1c2cec|f662683dfc8c3a1a+8032f74e33ebcad0+7d6c05abd13d3b81+d842fd5c8b1c2cec",
  "a74ce2fd03a04773+4dec36c67b9627f4+7d6c05abd13d3b81+8e5cd88d471f8119|d8cabe19b53909a6+70a4f9733ccee36a+7d6c05abd13d3b81+8e5cd88d471f8119",
  "5a2a7697cf8272d7+18c7d5859982dec3+7d6c05abd13d3b81+6e3fe72102eeaaa4|367832e2ab5a62e1+3a675332910005dc+7d6c05abd13d3b81+6e3fe72102eeaaa4",
  "bf6598aad061d1ab+4b42ebaef2764e86+7d6c05abd13d3b81+3a1231538300118f|9c3243db41ed3652+7c50e4f37d4ce887+7d6c05abd13d3b81+3a1231538300118f",
  "27208494829d80fc+4828ce769b38442a+7d6c05abd13d3b81+87eeee36351271a7|e1d9544e2c23cc35+02863c1c53d95f85+7d6c05abd13d3b81+87eeee36351271a7",
  "81429b0e8363189b+514407a3dd58b527+7d6c05abd13d3b81+0fc37273690c34b5|de7fc51ff1d3705a+3b40a97c092282bb+7d6c05abd13d3b81+0fc37273690c34b5",
  "52f7e72aa9c1185a+6a09ff27b12998d2+7d6c05abd13d3b81+ab10acb278ce6ab6|d49f6e424c316dac+a3ebf0ebebeb4b6a+7d6c05abd13d3b81+ab10acb278ce6ab6",
  "d45cc8a7f84fe67d+6a89c226a0aae04f+7d6c05abd13d3b81+512eb6b373f1a853|a9790a8d6c4f2864+9389532cd2bf74b5+7d6c05abd13d3b81+512eb6b373f1a853",
  "93729c64e8acec47+d7c4883078c62aa3+7d6c05abd13d3b81+e83fdb7e3f57ae5d|a35eaffbb6df66e2+9336fb3108b601ae+7d6c05abd13d3b81+e83fdb7e3f57ae5d",
  "3e3a00e0a8a8284d+b5c150c9342857c4+7d6c05abd13d3b81+b2af0e7ff6c78fc1|68d7c9894a12d92f+f8fcabc1baf3226d+7d6c05abd13d3b81+b2af0e7ff6c78fc1"
 ],
 "residual/0/controls": [
  [
   "0e483095192858b8+33759e3567b7680d+7d6c05abd13d3b81+7b52e5dea6d174d2",
   "da38ed83d9827d8f+831c7557d2805c03+7d6c05abd13d3b81+5dc187c663b5365a"
  ],
  [
   "2637496eb88cc5f2+d525100d69b38a9a+7d6c05abd13d3b81+eaeef59bc21fbc9f",
   "dce051893efb654d+21b827723a006054+7d6c05abd13d3b81+1fc1ebeba66b0e69"
  ],
  [
   "030352b56cecc15c+8f08c6cef9491798+7d6c05abd13d3b81+0dd51a657bfbe795",
   "5e83ecc510a7915c+24648b6bd4d54af0+7d6c05abd13d3b81+7ac66ddc67880d60"
  ],
  [
   "227fbe7153a1da40+242311dedd66fa7c+7d6c05abd13d3b81+6f815caaea030ebf",
   "926a5e83670cd3f8+973ec11592a15f12+7d6c05abd13d3b81+63a1b655f5a972d1"
  ],
  [
   "84a9e1754e8e331c+109c0a7de0ed9d4a+7d6c05abd13d3b81+60f8811e2bf21ed5",
   "f4fa3dbb8504c4da+b594fd95710284bc+7d6c05abd13d3b81+98d1e874f9ceb372"
  ],
  [
   "1d19c2e54550e2c8+a9618e1a9c0e3df3+7d6c05abd13d3b81+e6f4b647534d3f7d",
   "fc58dc02ea68ccdd+b41fc0862732567a+7d6c05abd13d3b81+91213fd7e97cecfb"
  ],
  [
   "0e00f22891ac9d78+6c7977850875c558+7d6c05abd13d3b81+22eb92bd75cb9cdd",
   "b28afcb6f0696d35+daf963919375b172+7d6c05abd13d3b81+7c815a680870cdcc"
  ],
  [
   "b559c479ba2866ff+da05225d55ce63ac+7d6c05abd13d3b81+2650222bddad0275",
   "f49045d6036479da+eef58ed671f439dd+7d6c05abd13d3b81+1e9b89707bc70e5c"
  ],
  [
   "4c9102e6224652b3+815a3f9dc14d0e3c+7d6c05abd13d3b81+bcfd762b8bc14026",
   "b341c3e8667a2e6f+1be0cec27bc92951+7d6c05abd13d3b81+7942f7b2622647e5"
  ],
  [
   "28659323091a1bf2+e560ff60b4c5c871+7d6c05abd13d3b81+15c090ef90ef5b13",
   "21e100b09471d1ff+2b1f3308b391e9d4+7d6c05abd13d3b81+2cdd4c9e0f7d83c9"
  ]
 ],
 "residual/1/pairs": [
  "efbf206d1c74f7f8+00062a2fb0c0923b+7d6c05abd13d3b81+1677571629ff6bb5|f1ba4523eee3df9c+e00437fa961e77f4+7d6c05abd13d3b81+1677571629ff6bb5",
  "7589c09a172c18cb+01a63fd80255716b+7d6c05abd13d3b81+a7896502a0a153fd|74e8fcde9bfa767a+a2ad02635dfb231e+7d6c05abd13d3b81+a7896502a0a153fd",
  "6b95576502ed24a3+4281a3af02fa871c+7d6c05abd13d3b81+8818f7f61d2e5d2b|700dfc0d4e9ba6bd+478c560f9efaa429+7d6c05abd13d3b81+8818f7f61d2e5d2b",
  "39be7a63af0ea0ee+16fbc4be5470dbc0+7d6c05abd13d3b81+8c413812e4063dde|ae2737824d8ad12b+e8da5aa7f6c0e740+7d6c05abd13d3b81+8c413812e4063dde",
  "80a5ec57e2bfe76d+a029106c8c65c78a+7d6c05abd13d3b81+ea4c48ef724b807e|84cd56da6c9ef4d7+fa2a0fe052ed7534+7d6c05abd13d3b81+ea4c48ef724b807e",
  "aa6c1c3179a29bbf+cad45cf453b74d09+7d6c05abd13d3b81+65fd2e98cadd0035|6ddb11ddfac7f136+ff08ffdf1237632e+7d6c05abd13d3b81+65fd2e98cadd0035",
  "07b251031c9bc8ac+cadd4883eb939fab+7d6c05abd13d3b81+5f02151141171de4|ccd2dbbc1cb24a88+ea5c27115da36bbd+7d6c05abd13d3b81+5f02151141171de4",
  "0f1a37a0c3a74488+d212a68a4df707f6+7d6c05abd13d3b81+935649fcd9f35830|0dddc4246f03564b+8dd61ca70abf6e80+7d6c05abd13d3b81+935649fcd9f35830",
  "15b1c99cb3c843c0+562085d5364e97bd+7d6c05abd13d3b81+b2e688cd57eec27e|7ec62ca29622ec3e+924eaf647cd214f6+7d6c05abd13d3b81+b2e688cd57eec27e",
  "330594f1e6a799a5+a587633199586443+7d6c05abd13d3b81+a57e3954485c6ec9|69017efb48154bf5+1afcfdb2d8095f8f+7d6c05abd13d3b81+a57e3954485c6ec9"
 ],
 "residual/1/controls": [
  [
   "4b7f8f79de2180be+2f37c3828829f149+7d6c05abd13d3b81+0cf0923f251828a2",
   "242f2675149c7691+f88a44b5e3d45a65+7d6c05abd13d3b81+6b8259163f9aee48"
  ],
  [
   "00da75266c091dbe+177b37920e5af252+7d6c05abd13d3b81+9b7df4c54fe8be27",
   "34ac4d577c5e4c18+aa3c81c5278fb1c7+7d6c05abd13d3b81+c5d850bf7a177848"
  ],
  [
   "0c6e5d51e9f582c2+49e18180b7379371+7d6c05abd13d3b81+645b41c27849648b",
   "96d1de84347d9643+b019db91bc677468+7d6c05abd13d3b81+8a42f1c1973ea73c"
  ],
  [
   "185fa8e9dbe1389a+c6963a56350edd45+7d6c05abd13d3b81+6798f12c8bc3bd54",
   "e309eec2e71ceb07+4d461c54bd11b21b+7d6c05abd13d3b81+3b712fcf8744a68c"
  ],
  [
   "ded30f5741279101+cf79903d3954ead7+7d6c05abd13d3b81+e262d6151579bf31",
   "d9e15c1635b4f35c+f9d068d068f6f594+7d6c05abd13d3b81+eab66a2b7e3aa789"
  ],
  [
   "ddee542fb6e6b8d6+b8cd2c8fe9a24e1e+7d6c05abd13d3b81+d28cade8ee9a172d",
   "7836eb0c701fc8ef+53ad4b5dd4087746+7d6c05abd13d3b81+4a8a43317424702c"
  ],
  [
   "771e24232bd2da58+999ff3136618051d+7d6c05abd13d3b81+f41a18e1798b500d",
   "3a4cc563fe58d657+4796a02b7ac05c33+7d6c05abd13d3b81+c45f8e38ed631117"
  ],
  [
   "8fc75ec7a41187b1+1ba4308ed06a8c8e+7d6c05abd13d3b81+85f32f911d6b542e",
   "ee66e3629c38332f+5d53943521ff6ef4+7d6c05abd13d3b81+cbabff463c6b330d"
  ],
  [
   "310d0eba8190595a+0c5678b0a6cf1f80+7d6c05abd13d3b81+2ae97a7f9c670ba7",
   "d8e2a8dfd9fc30ec+0ccb4baa7039ea84+7d6c05abd13d3b81+cae3d5708ebfd711"
  ],
  [
   "7c34d924882fb762+f38c406d4b722b3f+7d6c05abd13d3b81+0c8a566ca2030be5",
   "82cb6958c6bbc7b7+ed2f525558cc8b89+7d6c05abd13d3b81+9aa5a96f468cbdaa"
  ]
 ],
 "residual/2/pairs": [
  "55d320ce08163ecf+321b7c3eea37d6f0+7d6c05abd13d3b81+349694e61954e541|790cab93a9c2e535+cedda9996908ad84+7d6c05abd13d3b81+349694e61954e541",
  "0a1de4d6cdc7cc5a+6e8cd2911a5b455c+7d6c05abd13d3b81+a8778b2d3a7d894a|0b3a070f293bd5c3+6f4a738446f878a3+7d6c05abd13d3b81+a8778b2d3a7d894a",
  "25ba577f833b114c+18fb5f66825b9f81+7d6c05abd13d3b81+a2a168371a02222e|ec196cca3f9d823f+3fdbec2cb610b829+7d6c05abd13d3b81+a2a168371a02222e",
  "25edc90139ac1faa+5750905c279aa201+7d6c05abd13d3b81+bdc8b2cc5895ba6c|f41231c4d3a4505b+e4389976cdb1035c+7d6c05abd13d3b81+bdc8b2cc5895ba6c",
  "8f6fd2d0bc82e4f7+6985fbce6b560d08+7d6c05abd13d3b81+d57bf18d26854789|ddd3f232841080b5+cdb927a81113d04e+7d6c05abd13d3b81+d57bf18d26854789",
  "953dea855e45ff93+5b158689d10b2fa1+7d6c05abd13d3b81+9530b114ff2fde3f|cef97fd389b43692+3c53c599c1e8b298+7d6c05abd13d3b81+9530b114ff2fde3f",
  "5fcf6cdbab790cfa+c7b700d46b3a04ed+7d6c05abd13d3b81+5e922090c284e8f8|35d08299f2317896+82eb726a9ef5545a+7d6c05abd13d3b81+5e922090c284e8f8",
  "5a87098c7b90fa95+9eb3d05d4e76acf5+7d6c05abd13d3b81+6205d5f6f066f037|cdc384f116fd908e+d08c2115950dd73d+7d6c05abd13d3b81+6205d5f6f066f037",
  "1076d80274048abf+c53b83b8c3d5dc88+7d6c05abd13d3b81+8865e5c4eb1ddb92|593a67ce33a32e87+f7a7f5e2e8cb2297+7d6c05abd13d3b81+8865e5c4eb1ddb92",
  "4a25390e3e9f4205+59b05b7c74a84399+7d6c05abd13d3b81+937df3adb22aab3e|921c60a58efca7d0+8424a46dce6e8045+7d6c05abd13d3b81+937df3adb22aab3e"
 ],
 "residual/2/controls": [
  [
   "5382aefba870be07+4d410c9770aa8df2+7d6c05abd13d3b81+a7ae6435612b5a93",
   "5dba8a3d109cb484+02e37647edd7d920+7d6c05abd13d3b81+c7aebb653b8ff643"
  ],
  [
   "9db64ecc2f26cd91+337aed207558cbb7+7d6c05abd13d3b81+042442136ff9b260",
   "d214f62217facea3+b92ed5aec8326517+7d6c05abd13d3b81+d916f4f164ec09bc"
  ],
  [
   "1ffa3ef48470ffad+04535211b5146623+7d6c05abd13d3b81+a11fd0e977f4d3f5",
   "704dd1aaaef1f1d2+8219438a03f66480+7d6c05abd13d3b81+b4188be8f1eeba0d"
  ],
  [
   "8bb25f7e55165ccb+cdbf29a1d8a0c283+7d6c05abd13d3b81+2b68781851f94384",
   "77cdd81d64d4dcf2+5fd84e780a3d32ce+7d6c05abd13d3b81+14cf505143545c62"
  ],
  [
   "c4a60b7c2fae49c0+e99208720fa8daa0+7d6c05abd13d3b81+f6e04a9d0754edcf",
   "463989b98409d881+b5c2492d7b9e6fab+7d6c05abd13d3b81+45af161b79b93246"
  ],
  [
   "306bab8969c3540b+3d06b83a54162b60+7d6c05abd13d3b81+71244d96fee2c971",
   "6197f8a8aa41c230+00a3ced4c282f337+7d6c05abd13d3b81+7ed1e1865e63fa9b"
  ],
  [
   "303a22e3079b21b3+80ad0b1acdb33860+7d6c05abd13d3b81+f1e61a73e76f3ab8",
   "1571b2db286963d7+a2f826858bc9236e+7d6c05abd13d3b81+3b80d96832c95677"
  ],
  [
   "12bfd9493584d3f4+006db15075ef2f8d+7d6c05abd13d3b81+c7263eb8423bf069",
   "8e3ddf3a391f110e+ac7d0a5dcd809670+7d6c05abd13d3b81+c5320c67f3301c6b"
  ],
  [
   "a8f7ed6f6788acc2+ff23e371dc0dd87d+7d6c05abd13d3b81+5c4f861398de3f79",
   "fc72cd170760dc19+66645adc6e48b6bf+7d6c05abd13d3b81+34e175a9c05a5fad"
  ],
  [
   "a9e08d74d0512ce2+623aa53aa5d758d7+7d6c05abd13d3b81+aa10e5389473a14c",
   "29406a62f278e2a4+52950f6deb9dee2d+7d6c05abd13d3b81+521eaa3c351e4e24"
  ]
 ],
 "activation/0/pairs": [
  "2c0f6a207fad5391+281dab314b6f738a+3ea605dba1482acd+6584698db2398d74+4d5e53a01d7c98ca+544bfdcaec2f5140+e40c92334c4c9941|2c0f6a207fad5391+281dab314b6f738a+3ea605dba1482acd+6584698db2398d74+f4be843f5c8b7e02+6ca8662cccbee581+4c8460dd377dd3d6",
  "c11c0ca8448f2bcb+5d38bbfbc259f7ed+3ea605dba1482acd+43ec3427649bbf76+e3fa13fc6b6cb937+91e99dbec0e17daf+2b5407ad84001d6d|c11c0ca8448f2bcb+5d38bbfbc259f7ed+3ea605dba1482acd+43ec3427649bbf76+2ab14ed89864ac89+3bd025b89356e31f+49eddb13b39cb742"
 ],
 "activation/0/controls": [
  [
   "9e5ca704104323f7+d5d854a8bac34b42+3ea605dba1482acd+02861de4170a6778+fb366ad0c8506f5f+d2f6451dd7658e27+4c8460dd377dd3d6",
   "3361ee23241a7909+c652d6e6886c1956+3ea605dba1482acd+52c94ef8d7b102c8+412dd961f0b03606+c0b28bef20d6d2dc+4c8460dd377dd3d6"
  ],
  [
   "f87fe5e16b5dacba+2a57880481d434eb+3ea605dba1482acd+ddb4de04c2573aff+16ca848b9c80bbdd+1b27216cc18f1d01+49eddb13b39cb742",
   "e56579d141ed53a2+b21c860bbc15b4b5+3ea605dba1482acd+cb6a0e43930d1b60+710d969c1a03f49e+1ff6060f007d27a5+49eddb13b39cb742"
  ]
 ]
}""")


# ---------------------------------------------------------------------------
# the frozen registry satisfies the DSL
# ---------------------------------------------------------------------------


def test_frozen_registry_satisfies_the_dsl_schema() -> None:
    assert G.registry_problems(REGISTRY, P) == []
    ids = [entry["familyId"] for entry in REGISTRY["families"]]
    assert ids == ["period24", "cross-lag3", "nonlinear", "activation", "horizon-range", "level-bias"]
    origins = {entry["familyId"]: entry["origin"] for entry in REGISTRY["families"]}
    assert origins["horizon-range"] == "generated"
    assert origins["level-bias"] == "generated"
    assert all(origins[f] == "seed" for f in ("period24", "cross-lag3", "nonlinear", "activation"))


def test_seed_families_keep_their_historical_seed_slots() -> None:
    """The migration must not move a child-seed slot: the legacy layout was
    family-major pairs at 2*f+p and controls at 6 + position*2 + c."""
    expected_pairs = {"period24": [0, 1], "cross-lag3": [2, 3], "nonlinear": [4, 5]}
    for family_id, pair_seeds in expected_pairs.items():
        layout = family_entry(family_id)["seedLayout"]
        assert layout["pairSeeds"] == pair_seeds
        assert layout["controlSeeds"] == [
            [6 + 2 * position, 7 + 2 * position] for position in pair_seeds
        ]
    assert family_entry("activation")["seedLayout"] == {
        "netSeed": 0,
        "directionSeed": 1,
        "pairDataSeeds": [2, 3],
        "controlDataSeeds": [[4, 5], [6, 7]],
    }


def test_checked_in_proposals_match_the_generated_registry_entries() -> None:
    """The registry is the audit trail of the CLI admissions: the two checked-in
    proposal files must be exactly what was appended (DSL, witness, rationale)."""
    for family_id in ("horizon-range", "level-bias"):
        proposal = json.loads((_PROPOSALS_DIR / f"{family_id}.json").read_text(encoding="utf-8"))
        entry = family_entry(family_id)
        assert entry["generator"] == proposal["generator"]
        assert entry["witnessProbe"] == proposal["witnessProbe"]
        assert entry["rationale"] == proposal["rationale"]
        evidence = entry["gateEvidence"]
        assert evidence["witness"]["deterministic"] is True
        assert all(seed_data["complete"] for seed_data in evidence["witness"]["perSeed"].values())
        # Gate 2 with numbers: no registered witness completed the family.
        assert all(
            probe["completedAtSeeds"] == []
            for probe in evidence["specificity"]["probes"].values()
        )
        assert evidence["amplitude"]["maxStdChange"] < G.STD_CHANGE_BOUND
        assert evidence["amplitude"]["maxMeanShiftStd"] < G.MEAN_SHIFT_STD_BOUND
        assert evidence["reproducibility"]["bitwiseIdentical"] is True


# ---------------------------------------------------------------------------
# bitwise reproduction
# ---------------------------------------------------------------------------


def test_frozen_registry_reproduces_pre_pipeline_fixtures_bitwise() -> None:
    for kind, seeds, n_legacy_pairs in (("residual", (0, 1, 2), 6), ("activation", (0,), 2)):
        for seed in seeds:
            pairs = P.make_fixtures(kind, seed=seed)
            controls = P.make_controls(kind, seed=seed)
            expected_pairs = _EXPECTED_DIGESTS[f"{kind}/{seed}/pairs"][:n_legacy_pairs]
            expected_controls = _EXPECTED_DIGESTS[f"{kind}/{seed}/controls"][:n_legacy_pairs]
            assert len(pairs) == 2 * len(P._family_ids(kind))
            # the legacy families come first in registry order
            got_pairs = [
                fixture_digest(planted) + "|" + fixture_digest(clean)
                for planted, clean in pairs[:n_legacy_pairs]
            ]
            got_controls = [
                [fixture_digest(control) for control in pair]
                for pair in controls[:n_legacy_pairs]
            ]
            assert got_pairs == expected_pairs
            assert got_controls == expected_controls


def test_generated_families_are_bitwise_reproducible() -> None:
    for family_id in ("horizon-range", "level-bias"):
        result = G.gate_reproducibility(P, family_entry(family_id), seeds=(0, 1))
        assert result["passed"], result["report"]


# ---------------------------------------------------------------------------
# the DSL schema rejects bad injections before any gate runs
# ---------------------------------------------------------------------------


def horizon_proposal(**overrides: Any) -> dict[str, Any]:
    proposal = json.loads((_PROPOSALS_DIR / "horizon-range.json").read_text(encoding="utf-8"))
    for key, value in overrides.items():
        proposal[key] = value
    return proposal


def _first_problem(proposal: dict[str, Any]) -> str:
    return "; ".join(G.proposal_problems(proposal, REGISTRY, P))


def test_zero_amplitude_is_rejected() -> None:
    proposal = horizon_proposal()
    proposal["generator"]["injection"]["amplitudeFraction"] = 0.0
    assert "amplitudeFraction must be a number in (0, 3]" in _first_problem(proposal)


def test_stationarity_breaking_clean_process_is_rejected() -> None:
    proposal = horizon_proposal()
    proposal["generator"]["clean"]["channels"][0]["phi"] = 1.0
    assert "phi must be in (-1, 1) for stationarity" in _first_problem(proposal)


def test_unknown_injection_type_is_rejected() -> None:
    proposal = horizon_proposal()
    proposal["generator"]["injection"] = {
        "type": "arbitrary_wave", "channel": 0, "amplitudeFraction": 0.5
    }
    assert "injection type must be one of" in _first_problem(proposal)


def test_seed_handling_is_not_the_proposals_to_control() -> None:
    # The pipeline assigns child-seed slots; a proposal carrying seed keys
    # (or a seedLayout of its own) falls outside the closed DSL.
    proposal = horizon_proposal()
    proposal["generator"]["seed"] = 12345
    assert "unknown keys ['seed']" in _first_problem(proposal)
    proposal = horizon_proposal()
    proposal["seedLayout"] = {"pairSeeds": [0, 1], "controlSeeds": [[2, 3], [4, 5]]}
    assert "unknown keys ['seedLayout']" in _first_problem(proposal)


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda p: p.update(familyId="Bad Id!"), "familyId must match"),
        (lambda p: p.update(familyId="period24"), "already in the registry"),
        (lambda p: p["witnessProbe"]["proposal"].pop("signalKey"), "missing key 'signalKey'"),
        (
            lambda p: p["witnessProbe"]["proposal"].update(expectedDirection="sideways"),
            "expectedDirection must be 'higher' or 'lower'",
        ),
    ],
)
def test_more_schema_rejections(mutation: Any, expected: str) -> None:
    proposal = horizon_proposal()
    mutation(proposal)
    assert expected in _first_problem(proposal)


# ---------------------------------------------------------------------------
# the gates, one job each
# ---------------------------------------------------------------------------


def test_amplitude_gate_rejects_a_mean_shifting_injection() -> None:
    """cycles=0.05 is a near-constant drift: the whole offset lands in the
    marginal mean and gate 3 must reject it with numbers (this is the
    level-bias design's failed first iteration)."""
    entry = copy.deepcopy(family_entry("level-bias"))
    entry["familyId"] = "level-bias-mean-shift"
    entry["generator"]["injection"]["cycles"] = 0.05
    result = G.gate_amplitude(P, entry, seeds=(0,))
    assert not result["passed"]
    assert "max mean shift" in result["report"]
    assert result["evidence"]["maxMeanShiftStd"] >= G.MEAN_SHIFT_STD_BOUND


def test_witness_gate_rejects_a_nondeterministic_witness() -> None:
    """The witness completes its family but sprinkles unseeded randomness into
    the signalKey: gate 1's determinism rerun must catch it."""
    entry = copy.deepcopy(family_entry("horizon-range"))
    entry["familyId"] = "horizon-range-nondeterministic"
    jittered = entry["witnessProbe"]["source"].replace(
        "return {\"horizonRatio\": num / den}",
        "return {\"horizonRatio\": (num / den) * (1.0 + 1e-9 * np.random.random())}",
    )
    assert jittered != entry["witnessProbe"]["source"]  # the mutation took
    entry["witnessProbe"]["source"] = jittered
    result = G.gate_witness(P, entry, seeds=(0,))
    assert not result["passed"]
    assert "deterministic" in result["report"]
    # the jitter is tiny: the family itself still completes — the failure is
    # the determinism rerun, not the pair rule
    assert result["evidence"]["perSeed"][0]["complete"] is True


def test_specificity_gate_rejects_a_design_the_band_probe_sees() -> None:
    """A period-24 sinusoid injection is NOT a new family: the registered
    band-24 witness completes it at every seed, so gate 2 rejects the proposal
    (the horizon-range design's failed first iteration)."""
    entry = copy.deepcopy(family_entry("horizon-range"))
    entry["familyId"] = "period24-clone"
    entry["generator"]["injection"] = {
        "type": "sinusoid", "channel": 0, "period": 24, "amplitudeFraction": 0.75,
    }
    entry["seedLayout"] = G.assign_seed_slots(REGISTRY, "residual")
    seed_registry = {
        "version": 1,
        "families": [e for e in REGISTRY["families"] if e["origin"] == "seed" and e["kind"] == "residual"],
    }
    result = G.gate_specificity(P, entry, seed_registry, seeds=(0,))
    assert not result["passed"]
    assert result["evidence"]["probes"]["band24-energy"]["completedAtSeeds"] == [0]


def test_admit_rejects_a_bad_proposal_without_touching_the_registry(tmp_path: Path) -> None:
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(REGISTRY, indent=2), encoding="utf-8")
    proposal = horizon_proposal()
    proposal["generator"]["injection"]["amplitudeFraction"] = 0.0
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")
    assert G.admit(proposal_path, registry_path, seeds=(0,)) == 1
    assert json.loads(registry_path.read_text(encoding="utf-8")) == REGISTRY  # untouched


def test_admit_rejects_a_duplicate_design_then_appends_a_new_one(tmp_path: Path) -> None:
    """The full CLI pipeline on the checked-in horizon-range proposal, against
    a registry copy.  Re-proposing the horizon design under a fresh id is
    REJECTED at gate 2 — the registered horizon-ratio witness completes the
    clone, because a family IS its witness signature and the registry must
    not hold two of the same.  A registry that has no residual families yet
    admits it: all four gates, then the append with the gate evidence."""
    proposal = horizon_proposal()
    proposal["familyId"] = "horizon-range-admission-test"
    proposal_path = tmp_path / "proposal.json"
    proposal_path.write_text(json.dumps(proposal), encoding="utf-8")

    # (a) duplicate design vs the full registry: gate 2 rejects, no append
    full = tmp_path / "registry-full.json"
    full.write_text(json.dumps(REGISTRY, indent=2), encoding="utf-8")
    assert G.admit(proposal_path, full, seeds=(0,)) == 1
    assert json.loads(full.read_text(encoding="utf-8")) == REGISTRY  # untouched

    # (b) the same proposal against a registry with no residual families:
    # gates 1, 3, 4 all pass (specificity is vacuous) and the family is
    # appended with its assigned slots and evidence
    fresh = tmp_path / "registry-fresh.json"
    fresh.write_text(
        json.dumps({"version": 1, "families": [family_entry("activation")]}, indent=2),
        encoding="utf-8",
    )
    assert G.admit(proposal_path, fresh, seeds=(0,)) == 0
    appended = json.loads(fresh.read_text(encoding="utf-8"))
    assert len(appended["families"]) == 2
    entry = appended["families"][-1]
    assert entry["familyId"] == "horizon-range-admission-test"
    assert entry["origin"] == "generated"
    assert entry["seedLayout"] == {"pairSeeds": [0, 1], "controlSeeds": [[2, 3], [4, 5]]}
    assert entry["gateEvidence"]["witness"]["perSeed"]["0"]["complete"] is True
    assert entry["gateEvidence"]["witness"]["deterministic"] is True
    assert entry["gateEvidence"]["reproducibility"]["bitwiseIdentical"] is True
    # and the REAL registry was not touched by the test admission
    assert len(REGISTRY["families"]) == 6
