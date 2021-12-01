@rmax = 100
@szmin = 8
@szmax = 20
@probtype = [:uniform, :direct_dist, :inverse_dist]
@feetype = [:direct_prob, :inverse_prob]

@feestep = 0.1
@fee = (2.5..3.5).step(@feestep).to_a

@ustep = 0.05
@pstep = 0.25
@uprob = (0.05..0.6).step(@ustep).to_a
@pprob = (0.25..0.5).step(@pstep).to_a

def get_pts(n)
    pts = [{x: 0, y: 0, r: 0}]

    n.times do
        r = @rmax * Math.sqrt(Random.rand)
        theta = 2 * Math::PI * Random.rand
        x = r * Math.cos(theta)
        y = r * Math.sin(theta)

        pts << {x: x, y: y, r: r}
    end

    return pts
end

def get_probs(probt, prob, pts)
    if probt == :uniform
        return get_uprobs(prob, pts.size)
    else
        return get_prop_probs(probt, prob, pts)
    end
end

def get_uprobs(prob, n)
    probs = [0]

    n.times do
        probs << prob + Random.rand(-@ustep..@ustep)
    end

    return probs
end

def get_prop_probs(probt, prob, pts)
    probs = [0]

    pts.each do |pt|
        gamma = (pt[:r].to_f / @rmax.to_f)
        gamma = 1.0 - gamma if probt == :inverse_dist
        probs << prob + Random.rand(-@pstep..@pstep) * gamma
    end

    return probs
end

def get_fees(feet, fee, probs)
    m, mm = probs.minmax
    fees = [0]

    probs.each do |pr|
        fac = (pr - m).to_f / (mm - m).to_f
        fac = fac.clamp(0.01, 0.99)

        if feet == :direct_prob
            fees << fee * @rmax * fac / probs.size.to_f
        else
            fees << fee * @rmax * (1.0 - fac) / probs.size.to_f
        end
    end

    return fees
end

def print_instance(pts, probt, prob, probs, feet, fee, fees)
    sz = pts.size
    name = "sz-#{sz}-prob_type-#{probt}-prob-#{'%.2f' % prob}-fee_type-#{feet}-fee-#{'%.1f' % fee}"

    instance = <<~EOF
        NAME : #{name}
        TYPE : TSP
        DIMENSION : #{sz}
        EDGE_WEIGHT_TYPE : EUC_2D
        NODE_COORD_SECTION
    EOF

    pts.each_with_index do |pt, i|
        instance << "#{i+1} #{'%.2f' % pt[:x]} #{'%.2f' % pt[:y]}\n"
    end

    instance << "ACCEPTED_PROBABILITIES\n"

    probs.each {|p| instance << "#{'%.2f' % p}\n"}

    instance << "OUTSOURCING_COSTS\n"

    fees.each {|m| instance << "#{'%.2f' % m}\n"}

    File.write("#{name}.txt", instance)
end

@szmin.upto(@szmax) do |size|
    @probtype.each do |probt|
        if probt == :uniform
            probs = @uprob
        else
            probs = @pprob
        end

        probs.each do |prob|
            @feetype.each do |feet|
                @fee.each do |fee|
                    pts = get_pts(size)
                    probs = get_probs(probt, prob, pts[1..])
                    fees = get_fees(feet, fee, probs[1..])

                    print_instance(pts, probt, prob, probs, feet, fee, fees)
                end
            end
        end
    end
end