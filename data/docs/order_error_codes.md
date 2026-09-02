# Order Error Code Reference

This page lists checkout and order-processing error codes agents will see in the order management system, along with the cause and the fix to walk the customer through.

## ERR-4032 — Payment authorization failed
Cause: the card issuer declined the authorization. Most often this is an expired card, an incorrect billing zip code, or insufficient funds. Less commonly it is triggered by the issuing bank's fraud filter flagging the transaction.
Fix: ask the customer to verify their card details (number, expiry, CVV, billing zip), update the payment method on file, and retry the charge. If it fails three times in a row, suggest an alternate payment method such as PayPal. Do not manually override or force-approve the charge under any circumstances.

## ERR-2210 — Address verification failed
Cause: the shipping address entered does not match the carrier's address database. Common triggers are a missing apartment/unit number, a misspelled street name, or an out-of-date zip code.
Fix: ask the customer to confirm the full address including unit or apartment number, then resubmit the order. If the address is a genuinely new address not yet in the carrier database (such as a brand-new subdivision), the order can be manually approved by a supervisor after the address is confirmed verbally.

## ERR-5091 — Inventory mismatch
Cause: the item went out of stock between the time it was added to the cart and the time checkout completed. This is more common during flash sales and limited restocks.
Fix: offer the customer a backorder with the restock date if known, or a full refund for that affected line item; the rest of the order still ships normally.

## ERR-1187 — Duplicate order detected
Cause: the same cart was submitted twice within 60 seconds, usually from a double-click on the checkout button or a slow page reload that the customer retried.
Fix: check the order history for a duplicate charge; if found, cancel and refund the duplicate order automatically. The customer should never be asked to dispute the charge with their bank for this — it should be resolved directly by support.

## ERR-3305 — Promo code rejected
Cause: the code is expired, has already been used by that account, or is being combined with a second promo code, which is not allowed.
Fix: confirm the code's expiry and usage rules with the customer, and remind them only one promo code is allowed per order.
